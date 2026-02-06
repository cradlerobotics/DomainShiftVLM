if [ $# -ne 1 ]; then
    echo "Usage: $0 <GPU_VRAM_SIZE>"
    exit 1
fi

GPU_VRAM_SIZE=$1

# If GPU_VRAM_SIZE is less than 12, we cannot run any VLMs
if [ "$GPU_VRAM_SIZE" -lt 12 ]; then
    echo "Insufficient VRAM size. At least 12GB is required to run VLMs."
    exit 1
fi

VLMS=("smolvlm2" "gemma3n" "blip2" "llama_vision" "mistral" "gemma" "qwen" "llava") # This can all fit provided that the VRAM size is at least 12

DATA=("real" "3d")

for D in "${DATA[@]}"; do
    echo "========RUNNING VIDEO SEGMENTATION========="
    python segmentation/segmentor.py --dataset $D --image data/$D/frame0000.png

    echo "========RUNNING CAPTIONING========="

    # Loop through each VLM and run the captioner
    for VLM in "${VLMS[@]}"; do
        echo "Running captioner for $VLM"
        python caption_data.py --captioner $VLM --dataset $D --tot_vram_gb $GPU_VRAM_SIZE
        python caption_data.py --captioner $VLM --use_masks --dataset $D --tot_vram_gb $GPU_VRAM_SIZE
    done
done
