import torch
import warnings
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .captioner import Captioner

class SmolVLM2Captioner(Captioner):
    """
    A captioner that uses Hugging Face's SmolVLM2-2.2B-Instruct model to generate captions for images.
    """
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device='cuda:0'):
        """
        Initialize the SmolVLM2 captioner.
        Args:
            model_id (str): Model ID from Hugging Face model hub
            device (str): Device to use for model inference
        """
        self.model_id = model_id
        self.processor = None
        self.model = None
        super().__init__(device)
        self._init_models()

    def _init_models(self):
        """Initialize the processor and model"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    # _attn_implementation="flash_attention_2"
                ).to(self.device)
            print(f"SmolVLM2Captioner initialized with model: {self.model_id}")
        except Exception as e:
            print(f"Error initializing SmolVLM2 models: {e}")
            raise

    def caption(self, imgs, user_prompt=None):
        """
        Caption the given images using the SmolVLM2 model.
        Args:
            imgs: List of images to caption (PIL.Image or numpy arrays)
            user_prompt (str): Custom prompt to use for captioning. 
                               If None, a default caption request will be used.
        Returns:
            List of captions for the images
        """
        if self.processor is None or self.model is None:
            self._init_models()
        if user_prompt is None:
            user_prompt = "Describe this image."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                captions = []
                for img in tqdm(imgs, desc="Captioning images (SmolVLM2)"):
                    try:
                        messages = [
                            {"role": "user", "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": user_prompt}
                            ]}
                        ]
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        ).to(self.model.device, dtype=torch.bfloat16)
                        input_len = inputs["input_ids"].shape[-1]
                        with torch.inference_mode():
                            generation = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
                            generation = generation[0][input_len:]
                        caption = self.processor.decode(generation, skip_special_tokens=True)
                        captions.append(caption)
                    except Exception as e:
                        print(f"Error generating caption (SmolVLM2): {e}")
                        captions.append("Error generating caption")
                return captions

    def stop(self):
        """Stop the captioner and release resources"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        print("SmolVLM2Captioner stopped and resources released.")
