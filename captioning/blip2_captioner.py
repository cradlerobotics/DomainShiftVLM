import os
import numpy as np
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken
import torch
import warnings
from transformers import BitsAndBytesConfig
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
from captioning.captioner import Captioner

class BLIP2Captioner(Captioner):
    def __init__(self, batch_size=20, device='cuda:0'):
        self.batch_size = batch_size
        self.device = device
        self.processor = None
        self.model = None
        super().__init__(device)
        self._init_models()

    def _init_models(self):
        """Initialize the processor and model"""
        try:
            # Change the quantization configuration to avoid float16/half precision
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", 
                    quantization_config=quantization_config, 
                    device_map='auto',
                    torch_dtype=torch.float32,  # Explicitly request float32
                )

            processor.num_query_tokens = model.config.num_query_tokens
            image_token = AddedToken("<image>", normalized=False, special=True)
            processor.tokenizer.add_tokens([image_token], special_tokens=True)

            model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
            model.config.image_token_index = len(processor.tokenizer) - 1

            self.processor = processor
            self.model = model
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise

    def caption(self, imgs, user_prompt=None):
        """Caption the given images"""
        # Initialize models if they haven't been initialized
        if self.processor is None or self.model is None:
            self._init_models()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                num_batches = int(np.ceil(len(imgs) / self.batch_size))
                captions = []
                print('Captioning objects with BLIP2 with batch size: ', self.batch_size)

                for batch_idx in tqdm(range(num_batches)):
                    batch_imgs = imgs[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
                    inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.model.device, torch.float16)
                    generated_ids = self.model.generate(**inputs)
                    batch_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    captions.extend(batch_captions)
                captions = [caption.strip() for caption in captions]
                return captions
            
    def stop(self):
        """Stop the captioner"""
        if self.model is not None:
            self.model = None
        if self.processor is not None:
            self.processor = None
        torch.cuda.empty_cache()

        

