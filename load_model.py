from transformers import AutoVideoProcessor, AutoModel

def load_jepa2(model_name, device):
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoVideoProcessor.from_pretrained(model_name)
    return model, processor


if __name__ == "__main__":
    hp_repo = "facebook/vjepa2-vitl-fpc64-256"
    load_jepa2(hp_repo, "cpu")