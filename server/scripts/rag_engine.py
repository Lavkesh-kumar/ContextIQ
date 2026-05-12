import numpy as np
from typing import List, Dict, Optional
from rdflib import Graph, Namespace, RDF, Literal, XSD, URIRef
from scripts.text_processor import TextProcessor
from scripts.vector_store import VectorStore
from openvino_genai import LLMPipeline


class RAGEngine:

    def __init__(self, model_path: str = "/app/model/phi-3-openvino", device: str = "CPU", max_tokens: int = 512):
        self.llm = LLMPipeline(model_path, device=device)
        self.max_tokens = max_tokens
        self.text_processor = TextProcessor()
        self.EX = Namespace("http://example.org/")

    def build_knowledge_graph(self, chunks: List[str]) -> Graph:
        """
        Build a knowledge graph from chunks.
        Each chunk is a node. Adjacent chunks are linked as neighbors
        so graph traversal can pull in surrounding context.
        """
        graph = Graph()
        graph.bind("ex", self.EX)

        for idx, chunk in enumerate(chunks):
            chunk_node = self.EX[f"Chunk_{idx}"]

            # Node type and index
            graph.add((chunk_node, RDF.type, self.EX.DocumentChunk))
            graph.add((chunk_node, self.EX.chunkIndex, Literal(idx, datatype=XSD.integer)))

            # Store full text as literal so we can retrieve it from graph
            graph.add((chunk_node, self.EX.hasText, Literal(chunk)))

            # Store a short preview
            preview = chunk[:100].replace("\n", " ")
            graph.add((chunk_node, self.EX.hasPreview, Literal(preview)))

            # Link to previous chunk (context window neighbor)
            if idx > 0:
                prev_node = self.EX[f"Chunk_{idx - 1}"]
                graph.add((chunk_node, self.EX.previousChunk, prev_node))
                graph.add((prev_node, self.EX.nextChunk, chunk_node))

        print(f"[KnowledgeGraph] Built graph with {len(chunks)} nodes, {len(graph)} triples.")
        return graph

    def get_graph_neighbors(self, graph: Graph, chunk_indices: List[int], chunks: List[str]) -> List[int]:
        """
        Given a list of retrieved chunk indices, use the knowledge graph
        to also pull in their immediate neighbors (prev + next chunks).
        Returns expanded sorted list of unique indices.
        """
        expanded = set(chunk_indices)

        for idx in chunk_indices:
            chunk_node = self.EX[f"Chunk_{idx}"]

            # Get next neighbor
            for neighbor in graph.objects(chunk_node, self.EX.nextChunk):
                neighbor_idx = self._node_to_index(graph, neighbor)
                if neighbor_idx is not None and neighbor_idx < len(chunks):
                    expanded.add(neighbor_idx)

            # Get previous neighbor
            for neighbor in graph.objects(chunk_node, self.EX.previousChunk):
                neighbor_idx = self._node_to_index(graph, neighbor)
                if neighbor_idx is not None and neighbor_idx >= 0:
                    expanded.add(neighbor_idx)

        # Return sorted so context is in document order
        return sorted(expanded)

    def _node_to_index(self, graph: Graph, node: URIRef) -> Optional[int]:
        """Extract chunk index integer from graph node."""
        for idx_val in graph.objects(node, self.EX.chunkIndex):
            try:
                return int(idx_val)
            except (ValueError, TypeError):
                return None
        return None

    def retrieve_relevant_docs(
        self,
        question: str,
        chunks: List[str],
        vector_store: VectorStore,
        graph: Graph,
        k: int = 3,
    ) -> List[str]:
        """
        Two-stage retrieval:
        1. Vector similarity search → top-k chunk indices
        2. Knowledge graph expansion → pull in neighboring chunks for richer context
        """
        if not chunks:
            return []

        # Stage 1: vector search
        query_embedding = self.text_processor.get_embeddings([question])
        distances, indices = vector_store.search(query_embedding, k)
        top_k_indices = indices[0].tolist()
        print(f"[VectorSearch] Top-k indices: {top_k_indices}, distances: {distances}")

        # Stage 2: graph neighbor expansion
        expanded_indices = self.get_graph_neighbors(graph, top_k_indices, chunks)
        print(f"[GraphExpansion] Expanded indices: {expanded_indices}")

        relevant_docs = [chunks[idx] for idx in expanded_indices]
        return relevant_docs

    def generate_answer(self, question: str, context: List[str]) -> str:

        context_text = "\n\n".join([f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(context)])

        prompt = f"""You are a helpful document question-answering assistant.
        
    Instructions:
    1. Answer the question based ONLY on the provided context
    2. Provide clear, well-structured answers
    3. If listing items (contacts, requirements, dates), include ALL relevant items from the context
    4. Do not include partial lists - if you start a list, complete it
    5. Use proper formatting: bullet points for lists, clear sections for different topics
    6. If the answer cannot be found in the context, say: "I cannot find the answer in the provided documents."
    7. Do not add disclaimers about "limited information" if you can answer from the context
    8. Be concise but complete - include all relevant details

Context:
{context_text}

Question: {question}

Answer:"""

        result = self.llm.generate(prompt, max_new_tokens=self.max_tokens)

        if isinstance(result, str):
            return result
        elif hasattr(result, 'texts'):
            return result.texts[0]
        elif isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                return result[0]['generated_text']
            return str(result[0])
        else:
            return str(result)

    def query(
        self,
        question: str,
        chunks: List[str],
        vector_store: VectorStore,
        graph: Graph,
        k: int = 3,
    ) -> Dict:
        """
        Full RAG pipeline:
        vector search → graph expansion → LLM answer generation
        """
        relevant_docs = self.retrieve_relevant_docs(
            question=question,
            chunks=chunks,
            vector_store=vector_store,
            graph=graph,
            k=k,
        )
        answer = self.generate_answer(question, relevant_docs)

        return {
            "answer": answer,
            "sources": relevant_docs,
            "num_sources": len(relevant_docs),
        }