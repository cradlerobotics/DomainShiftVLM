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

Download the models from Ollama:
```bash
bash download_from_ollama.sh
```

### Download SAM2 Checkpoint

```bash
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Captioning

During the first run, the captioning will download some models from HuggingFace and store them in `~/.cache/huggingface`.
```bash
bash run_all.sh
```

## Evaluation

Run the Jupyter notebook `first_frame_eval.ipynb`.