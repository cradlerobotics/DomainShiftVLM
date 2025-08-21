import torch
import os
from PIL import Image
import clip

class CLIPScorer:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def score(self, image_path, caption):
        """
        Compute CLIP score between an image file and a caption.
        image_path: path to the image file (already masked/cropped as needed)
        caption: string
        """

        # Process caption
        text_inputs = clip.tokenize([caption]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # Calculate similarity
            similarity = (text_features @ image_features.T).item()
        except Exception as e:
            raise ValueError(f"Error processing {image_path}: {e}")

        return similarity