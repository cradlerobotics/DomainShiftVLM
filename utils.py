import torch


def get_vram_info():
    """Return the total and available VRAM (in MB) if CUDA is available, else None."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_vram = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
        reserved_vram = torch.cuda.memory_reserved(device) // (1024 * 1024)
        allocated_vram = torch.cuda.memory_allocated(device) // (1024 * 1024)
        free_vram = total_vram - reserved_vram + (reserved_vram - allocated_vram)
        return {"total_vram_mb": total_vram, "free_vram_mb": free_vram}
    else:
        return None