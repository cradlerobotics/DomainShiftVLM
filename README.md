# DomainShiftVLM

This repository contains the data and the code for the paper "Evaluating the Robustness of Open-Source Vision-Language Models to Domain Shift in Object Captioning".

## Setup

```bash
conda create -y -n vlm python=3.10
conda activate vlm
pip install -r requirements.txt
```

Authenticate with Hugging Face to download required models:

```bash
huggingface-cli login
```
*(Create a token from your Hugging Face account if needed.)*

Download NLTK resources:

```python
import nltk
nltk.download('wordnet')
```

Install Ollama:

```bash
curl https://ollama.ai/install.sh | sh
```

Start Ollama:
```bash
ollama serve
```

Download the models from Ollama (specify your GPU's VRAM size: 12 or 24):
```bash
bash download_from_ollama.sh 12  # For 12GB VRAM
# OR
bash download_from_ollama.sh 24  # For 24GB VRAM
```

### Download SAM2 Checkpoint

```bash
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Captioning

During the first run, the captioning will download some models from HuggingFace and store them in `~/.cache/huggingface`.
```bash
bash run_all.sh 12  # For 12GB VRAM
# OR
bash run_all.sh 24  # For 24GB VRAM
```

## Evaluation

Run the Jupyter notebook `first_frame_eval.ipynb`.

## Codebase Structure

The repository is organized as follows:

```
DomainShiftVLM/
├── captioning/              # Caption generation modules
│   ├── captioner.py         # Abstract base class for all captioners
│   ├── blip2_captioner.py   # BLIP2 model implementation
│   ├── ollama_captioner.py  # Ollama-based VLM wrapper (llama, qwen, gemma, llava, mistral)
│   ├── gemma3n_captioner.py # Gemma 3N model implementation
│   └── smolvlm2_captioner.py # SmolVLM2 model implementation
│
├── segmentation/            # Segmentation modules
│   └── segmentor.py         # SAM2-based segmentation implementation
│
├── evaluation/              # Evaluation metrics
│   ├── benchmark.py         # Main evaluation pipeline with NLP metrics (CIDEr, BERTScore, ROUGE, GPTScore)
│   ├── clipscore.py         # CLIP-based evaluation
│   └── gptscore.py          # GPT-based scoring
│
├── data/                    # Dataset storage (3d/, real/)
├── output/                  # Output storage for segmentation and captioning results
├── checkpoints/             # Model checkpoints (SAM2)
│
├── caption_data.py          # Main script for running captioning pipeline
├── configs.json             # Configuration file for prompts and settings
├── ollama_map.json          # Mapping of VLM models to Ollama model names by VRAM size
├── utils.py                 # Utility functions (VRAM info)
├── run_all.sh               # Script to run full pipeline (segmentation + captioning)
├── download_from_ollama.sh  # Script to download Ollama models
├── first_frame_eval.ipynb   # Jupyter notebook for evaluation
└── requirements.txt         # Python dependencies
```

### Key Components

#### 1. Captioning Module (`captioning/`)
All captioners inherit from the abstract `Captioner` base class, which defines three key methods:
- `_init_models()`: Initialize the model and processor
- `caption(imgs, user_prompt=None)`: Generate captions for a list of images
- `stop()`: Clean up resources and free GPU memory

**Available Captioners:**
- **BLIP2** (`blip2_captioner.py`): Uses Salesforce/blip2-opt-2.7b with 8-bit quantization
- **Ollama VLMs** (`ollama_captioner.py`): Wrapper for Ollama-based models including:
  - llama_vision (Llama 3.2 Vision)
  - qwen (Qwen 2.5 Vision)
  - gemma (Gemma 3)
  - llava (LLaVA)
  - mistral (Mistral Small)
- **Gemma 3N** (`gemma3n_captioner.py`): Google's Gemma 3N model
- **SmolVLM2** (`smolvlm2_captioner.py`): Lightweight vision-language model

#### 2. Segmentation Module (`segmentation/`)
- Uses SAM2 (Segment Anything 2) for automatic object segmentation
- Generates masks and bounding boxes for objects in frames
- Includes mask cleaning and blob removal for better quality
- Outputs: segmentation masks, bounding boxes, and visualization images

#### 3. Evaluation Module (`evaluation/`)
- **NLP Metrics**: CIDEr, BERTScore, ROUGE-L, GPTScore
- **Vision Metrics**: CLIPScore for image-text alignment
- Benchmark tools for comparing model performance across domains

#### 4. Main Scripts
- **`caption_data.py`**: Main captioning pipeline that:
  - Loads segmentation results
  - Crops and optionally masks objects
  - Generates captions using selected VLM
  - Saves results to CSV
- **`run_all.sh`**: Automated pipeline for running segmentation and captioning on all VLMs
- **`download_from_ollama.sh`**: Downloads appropriate Ollama models based on VRAM size

### Configuration

#### `configs.json`
Contains prompts for each VLM model and device settings:
```json
{
  "general": {
    "device": "cuda"
  },
  "prompts": {
    "vlm": {
      "gemma": "Describe in detail the object...",
      ...
    }
  }
}
```

#### `ollama_map.json`
Maps VLM names to specific Ollama model variants based on available VRAM:
- 12GB VRAM: Smaller models (e.g., llama3.2-vision:11b, qwen2.5vl:7b)
- 24GB VRAM: Larger models (e.g., qwen2.5vl:32b, gemma3:27b)

### Data Organization

#### Input Data Structure
```
data/
├── 3d/
│   ├── frame0000.png
│   ├── frame0001.png
│   └── ...
└── real/
    ├── frame0000.png
    ├── frame0001.png
    └── ...
```

#### Output Data Structure
```
output/
├── 3d/
│   ├── sam2_segmentation/       # Segmentation visualizations
│   ├── sam2_tracking/           # Tracking data (masks, bboxes)
│   │   └── tracking_data.npz
│   └── caption/
│       ├── blip2/
│       │   ├── with_masks/
│       │   │   └── all_captions.csv
│       │   └── without_masks/
│       │       └── all_captions.csv
│       ├── gemma3n/
│       └── ...
└── real/
    └── (same structure as 3d/)
```

## Development Guidelines

### Adding a New VLM Model

To add a new vision-language model to the codebase:

1. **Create a new captioner class** in `captioning/`:
   ```python
   # captioning/your_model_captioner.py
   from captioning.captioner import Captioner
   
   class YourModelCaptioner(Captioner):
       def __init__(self, device='cuda:0'):
           super().__init__(device)
           self._init_models()
       
       def _init_models(self):
           # Initialize your model and processor
           pass
       
       def caption(self, imgs, user_prompt=None):
           # Generate captions for images
           # Return list of caption strings
           pass
       
       def stop(self):
           # Clean up resources
           pass
   ```

2. **Import the captioner** in `caption_data.py`:
   ```python
   from captioning.your_model_captioner import YourModelCaptioner
   ```

3. **Add to model selection** in `caption_data.py`:
   ```python
   def select_captioner(captioner_name, tot_vram_gb, device):
       if captioner_name == "your_model":
           return YourModelCaptioner(device=device)
       # ... existing cases
   ```

4. **Add prompt configuration** in `configs.json`:
   ```json
   "prompts": {
     "vlm": {
       "your_model": "Your prompt template here..."
     }
   }
   ```

5. **Update run script** in `run_all.sh`:
   ```bash
   VLMS=("smolvlm2" "gemma3n" "blip2" "your_model" ...)
   ```

### Adding a New Evaluation Metric

To add a new evaluation metric:

1. **Create a metric function** in `evaluation/benchmark.py` or a new file:
   ```python
   def calculate_your_metric(references, candidates):
       """Calculate your custom metric"""
       # Implementation
       return scores
   ```

2. **Integrate into benchmark**:
   ```python
   def calculate_nlp_metrics(ground_truth, predictions):
       results = {
           'cider': calculate_cider(...),
           'your_metric': calculate_your_metric(...)
       }
       return results
   ```

3. **Update evaluation notebook** to include your new metric.

### Code Style Guidelines

1. **Imports**: Standard library, third-party, then local imports
2. **Docstrings**: Use clear docstrings for classes and complex functions
3. **Error Handling**: Include try-except blocks for model loading and inference
4. **Resource Management**: Always implement `stop()` method to free GPU memory
5. **Batch Processing**: Use batch processing with tqdm for progress tracking
6. **Device Management**: Support both CUDA and CPU, default to CUDA

### Running Individual Components

#### Run Segmentation Only
```bash
python segmentation/segmentor.py --dataset real --image data/real/frame0000.png
```

#### Run Captioning with Specific Model
```bash
# Without masks
python caption_data.py --captioner blip2 --dataset real

# With masks
python caption_data.py --captioner gemma3n --use_masks --dataset 3d --tot_vram_gb 24
```

#### Run Specific VLM via Ollama
```bash
python caption_data.py --captioner llama_vision --dataset real --tot_vram_gb 12
```

### Testing Your Changes

1. **Test model loading**: Ensure your captioner initializes without errors
2. **Test inference**: Verify captions are generated correctly
3. **Test resource cleanup**: Confirm GPU memory is freed after `stop()`
4. **Test integration**: Run through full pipeline with your changes

### Common Issues and Solutions

1. **Out of Memory (OOM)**:
   - Reduce batch size in captioner
   - Use 8-bit quantization
   - Select smaller model variants

2. **Model Download Issues**:
   - Authenticate with HuggingFace: `huggingface-cli login`
   - For Ollama models: `ollama pull <model-name>`

3. **CUDA Compatibility**:
   - Install correct PyTorch version for your CUDA version
   - Check compatibility: `torch.cuda.is_available()`

4. **Segmentation Quality**:
   - Adjust `threshold_area` in `segmentor.py` for your dataset
   - Modify mask cleaning parameters for better results