import ollama
from tqdm import tqdm
import tempfile
import subprocess as sp
from .captioner import Captioner
from PIL import Image

class OllamaCaptioner(Captioner):
    """
    A captioner that uses Ollama to generate captions for images.
    """
    def __init__(self, model_name="gemma3:12b", device=None):
        """
        Initialize the Ollama captioner.
        
        Args:
            model_name (str): Name of the Ollama model to use
            device (str): Device is ignored as Ollama handles its own device assignment
        """
        self.model_name = model_name
        super().__init__(device)  # device is ignored for Ollama
        print(f"OllamaCaptioner initialized with model: {self.model_name}")

    def _init_models(self):
        pass  # Ollama models are initialized on demand
        
    def caption(self, imgs, user_prompt=None):
        """
        Caption the given images using the Ollama model.
        
        Args:
            imgs: List of images to caption
            user_prompt (str): Custom prompt to use for captioning
            
        Returns:
            List of captions for the images
        """
        captions = []
        print(f'Captioning images with Ollama model ({self.model_name})...')
        
        with tqdm(total=len(imgs)) as pbar:
            for img in imgs:
                try:
                    # Convert PIL Image to bytes
                    import io
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    messages=[{
                            'role': 'user',
                            'content': user_prompt,
                            'images': [img_byte_arr]
                    }]
                    
                    # Use the ollama Python package to send the request
                    response = ollama.chat(
                        model=self.model_name,
                        messages=messages,
                        options={
                            "num_predict": 20
                        }
                    )
                    
                    # Extract the response content
                    caption = response['message']['content']
                                        
                except Exception as e:
                    print(f"Error generating caption: {e}")
                    caption = "ERROR"
                captions.append(caption)
                pbar.update(1)
            pbar.close()
        return captions
    
    def stop(self):
        """
        Stop the captioner and release any resources.
        """
        print("Stopping OllamaCaptioner...")
        sp.Popen(["ollama", "stop", self.model_name])
