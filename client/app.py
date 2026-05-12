import streamlit as st
import requests

API_BASE_URL = "http://localhost:7860"

st.set_page_config(page_title="ContextIQ Document Q&A", layout="wide")

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "history" not in st.session_state:
    st.session_state.history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None


def main():

    st.markdown('<div class="main-header">ContextIQ Secure Document Q&A</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "docx", "pptx", "xlsx", "txt"]
        )

        st.markdown("---")
        st.subheader("Retrieval Settings")

        k_value = st.slider(
            "Number of document chunks",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant chunks to retrieve for answering questions"
        )

        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.rerun()

        if st.button("Reset Document"):
            st.session_state.doc_id = None
            st.session_state.doc_info = None
            st.session_state.history = []
            st.session_state.document_processed = False
            st.rerun()

        st.markdown("---")

        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=60)
            if resp.status_code == 200:
                st.success("API: Online ")
            else:
                st.error("API: Unhealthy ")
        except Exception:
            st.error("API: Unreachable ")

    if uploaded_file:

        if not st.session_state.document_processed:
            with st.spinner("Uploading and processing document..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/v1/documents",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=120,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.doc_id = data["doc_id"]
                        st.session_state.doc_info = data
                        st.session_state.document_processed = True
                        st.success(f"Document indexed! ({data['num_chunks']} chunks)")
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.Timeout:
                    st.error("Request timed out. The document may be too large.")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

        if st.session_state.doc_info:
            info = st.session_state.doc_info
            st.info(f"**{info['filename']}** | {info['num_chunks']} chunks | ID: `{info['doc_id']}`")

        st.markdown("---")
        st.subheader("Ask a Question")

        question = st.text_input(
            "Enter your question",
            placeholder="What is this document about?",
            key="question_input"
        )

        if st.button("🔍 Ask", type="primary") and question:
            if not question.strip():
                st.warning("Please enter a valid question.")
            elif not st.session_state.doc_id:
                st.warning("No document loaded. Please upload a document first.")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/v1/query",
                            json={
                                "doc_id": st.session_state.doc_id,
                                "question": question,
                                "k": k_value,
                            },
                            timeout=300,
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.history.append({
                                "question": question,
                                "answer": result["answer"],
                                "sources": result["sources"],
                            })
                            st.success("Answer generated!")

                        elif response.status_code == 404:
                            st.error("Document not found. Please re-upload your document.")
                            st.session_state.document_processed = False
                            st.session_state.doc_id = None

                        else:
                            st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")

                    except requests.exceptions.Timeout:
                        st.error("Query timed out. The model is taking too long on CPU.")
                    except Exception as e:
                        st.error(f"Error connecting to API: {e}")

        if st.session_state.history:
            st.markdown("---")
            st.subheader("Conversation History")

            for i, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Q{len(st.session_state.history)-i}: {item['question']}", expanded=(i == 0)):
                    st.markdown("**Answer:**")
                    st.write(item["answer"])

                    st.markdown("---")
                    st.markdown("**Retrieved Context:**")
                    for idx, src in enumerate(item["sources"], 1):
                        with st.expander(f"Chunk {idx}"):
                            st.text(src[:500] + "..." if len(src) > 500 else src)

    else:
        st.info("Upload a document from the sidebar to get started.")


if __name__ == "__main__":
    main()
