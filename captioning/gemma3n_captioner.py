import torch
import warnings
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from .captioner import Captioner

class Gemma3nCaptioner(Captioner):
    """
    A captioner that uses Hugging Face's Gemma-3n model to generate captions for images.
    """
    def __init__(self, model_id="google/gemma-3n-e4b-it", device='cuda:0'):
        """
        Initialize the HF captioner.
        
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
                self.model = Gemma3nForConditionalGeneration.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                ).eval()
                
            print(f"HFCaptioner initialized with model: {self.model_id}")
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise
        
    def caption(self, imgs, user_prompt):
        """
        Caption the given images using the Hugging Face model.
        
        Args:
            imgs: List of images to caption
            user_prompt (str): Custom prompt to use for captioning. 
                               If None, a default caption request will be used.
                
        Returns:
            List of captions for the images
        """
        # Initialize models if they haven't been initialized
        if self.processor is None or self.model is None:
            self._init_models()
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                captions = []
                
                for img in tqdm(imgs, desc="Captioning images"):
                    try:
                        # Create message with the image and prompt
                        messages = [
                            {"role": "user", "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": user_prompt}
                            ]}
                        ]
                        
                        # Process inputs
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        ).to(self.model.device)
                        
                        input_len = inputs["input_ids"].shape[-1]
                        
                        # Generate caption
                        with torch.inference_mode():
                            generation = self.model.generate(**inputs, max_new_tokens=1000, do_sample=False)
                            generation = generation[0][input_len:]
                            
                        caption = self.processor.decode(generation, skip_special_tokens=True)
                        captions.append(caption)
                        
                    except Exception as e:
                        print(f"Error generating caption: {e}")
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
        print("HFCaptioner stopped and resources released.")
