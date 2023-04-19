from huggingface_hub import snapshot_download

def main():
    local_dir = "./cat"
    snapshot_download("diffusers/cat_toy_example", local_dir=local_dir, repo_type="dataset",
                      ignore_patterns=".gitattributes")

if __name__ == "__main__":
    main()