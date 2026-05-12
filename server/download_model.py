from huggingface_hub import hf_hub_download
import os

model_id = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
save_directory = "./model/phi-3-openvino"

print("Downloading pre-converted OpenVINO Phi-3...\n")

os.makedirs(save_directory, exist_ok=True)

files_to_download = [
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "configuration_phi3.py",
    "chat_template.jinja",
]

try:
    for filename in files_to_download:
        print(f"Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=save_directory,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"Could not download {filename}: {e}")

    print(f"\n Model ready at: {save_directory}\n")

    files = os.listdir(save_directory)
    for file in sorted(files):
        print(f"  {file}")

except Exception as e:
    print(f"\n ERROR: {str(e)}")
