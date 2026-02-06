import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from pathlib import Path
import matplotlib.pyplot as plt
import os
import logging
import argparse

class Segmentor:
    def __init__(self, sam2_checkpoint, model_cfg, device="cuda"):
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.device = device
        self.output_dir = None
        self.logger = None

    def setup_logging(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(self.output_dir / "segmentation.log"),
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger()

    def get_bbox_from_mask(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        return [x_min, y_min, x_max, y_max]

    def visualize_frame_with_masks(self, frame, masks, save_path=None):
        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        cmap = plt.get_cmap('tab20')
        n_masks = len(masks)
        overlay = np.zeros_like(frame, dtype=np.float32)
        for i, mask in enumerate(reversed(masks)):
            color_idx = i % 20
            color = cmap(color_idx)[:3]
            mask_color = np.zeros((*frame.shape[:2], 3), dtype=np.float32)
            mask_color[mask > 0] = color
            alpha = 0.4
            mask_binary = mask > 0
            overlay[mask_binary] = (1-alpha) * overlay[mask_binary] + alpha * mask_color[mask_binary]
            bbox = self.get_bbox_from_mask(mask)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                plt.plot([x_min, x_max, x_max, x_min, x_min], 
                         [y_min, y_min, y_max, y_max, y_min], 
                         color=color, linewidth=2)
        plt.imshow(overlay, alpha=0.7)
        plt.title(f"Found {n_masks} objects", fontsize=14)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
        else:
            plt.show()

    def clean_masks(self, masks, dataset):
        threshold_area = 1500 if dataset == "3d" else 4000
        cleaned_masks = [mask for mask in masks if mask["area"] > threshold_area]
        if dataset != "3d":
            filtered_masks = []
            for i, mask in enumerate(cleaned_masks):
                bbox_i = mask["bbox"]
                contained = False
                for j, other_mask in enumerate(cleaned_masks):
                    if i != j:
                        bbox_j = other_mask["bbox"]
                        if (bbox_i[0] >= bbox_j[0] and 
                            bbox_i[1] >= bbox_j[1] and 
                            bbox_i[0] + bbox_i[2] <= bbox_j[0] + bbox_j[2] and 
                            bbox_i[1] + bbox_i[3] <= bbox_j[1] + bbox_j[3]):
                            contained = True
                            break
                if not contained:
                    filtered_masks.append(mask)
        else:
            filtered_masks = []
            for i, mask in enumerate(cleaned_masks):
                bbox_i = mask["bbox"]
                contains_other = False
                for j, other_mask in enumerate(cleaned_masks):
                    if i != j:
                        bbox_j = other_mask["bbox"]
                        if (bbox_i[0] <= bbox_j[0] and 
                            bbox_i[1] <= bbox_j[1] and 
                            bbox_i[0] + bbox_i[2] >= bbox_j[0] + bbox_j[2] and 
                            bbox_i[1] + bbox_i[3] >= bbox_j[1] + bbox_j[3]):
                            contains_other = True
                            break
                if not contains_other:
                    filtered_masks.append(mask)
        return filtered_masks

    def remove_small_blobs(self, masks):
        for i, mask in enumerate(masks):
            mask_data = mask["segmentation"]
            mask_data = (mask_data > 0).astype(np.uint8)
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_data)
            if num_labels > 2:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_component_idx = np.argmax(areas) + 1
                cleaned_mask = (labels_im == largest_component_idx).astype(np.uint8)
                mask["segmentation"] = cleaned_mask
            masks[i] = mask
        return masks

    def segment_image_with_sam2(self, image, dataset):
        print("Initializing SAM2 model for image segmentation...")
        sam2 = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2,
                                                    points_per_batch=32,
                                                    multimask_output=True)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        store_image = image_rgb.copy()
        print("Generating automatic segmentation masks...")
        masks = mask_generator.generate(image_rgb)
        is_real_data = dataset == "real"
        if is_real_data:
            if len(masks) > 0 and "segmentation" in masks[0]:
                background_mask = masks[0]["segmentation"]
                image_rgb[background_mask > 0] = 0
                masks = mask_generator.generate(image_rgb)
            else:
                print("No background mask found, using original image")
        masks = masks[1:]  # Skip the first mask which is usually the background
        masks = self.clean_masks(masks, dataset)
        if is_real_data:
            self.remove_small_blobs(masks)
        binary_masks = []
        for mask_data in masks:
            binary_masks.append(mask_data["segmentation"].astype(np.uint8))
        for i, mask in enumerate(binary_masks):
            mask_image = np.zeros_like(image_rgb)
            mask_image[mask > 0] = image_rgb[mask > 0]
            bbox = self.get_bbox_from_mask(mask)
            x_min, y_min, x_max, y_max = bbox
            mask_image = mask_image[y_min:y_max, x_min:x_max]
            cv2.imwrite(str(self.output_dir / f"mask_{i}.png"), cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
            print(f"Saved mask {i} to {self.output_dir / f'mask_{i}.png'}")
            cropped_image = store_image[y_min:y_max, x_min:x_max]
            cv2.imwrite(str(self.output_dir / f"cropped_{i}.png"), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
            print(f"Saved cropped image {i} to {self.output_dir / f'cropped_{i}.png'}")
            masks[i]["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
        print(f"SAM2 found {len(binary_masks)} masks")
        return masks, binary_masks, store_image

    def process_image(self, image_path, output_dir, dataset):
        self.setup_logging(output_dir)
        image = cv2.imread(image_path)
        if image is None:
            print("Error reading image")
            return {}, None
        height, width = image.shape[:2]
        print(f"Processing image of size {width}x{height}")
        if self.logger:
            self.logger.info("Starting segmentation with SAM2")
        masks, binary_masks, image_rgb = self.segment_image_with_sam2(image, dataset)
        self.visualize_frame_with_masks(
            image_rgb, 
            binary_masks, 
            save_path=str(self.output_dir / "sam2_segmentation.png")
        )
        bboxes = []
        for mask in masks:
            bboxes.append(mask["bbox"])
        print(f"Extracted {len(bboxes)} valid bounding boxes from SAM2")
        # Save mask data as npz
        mask_data_path = os.path.join(output_dir, "tracking_data.npz")
        try:
            np.savez_compressed(
                mask_data_path,
                num_objects=len(binary_masks),
                bboxes=np.array(bboxes),
                **{f"mask_{i}": m for i, m in enumerate(binary_masks)}
            )
            print(f"Saved mask data to {mask_data_path}")
        except Exception as e:
            print(f"Failed to save mask data: {str(e)}")
        return masks, binary_masks

def run_segmentation(dataset, image_path):    
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    output_dir = f"output/{dataset}/sam2_tracking"
    segmentor = Segmentor(sam2_checkpoint, model_cfg)
    masks, binary_masks = segmentor.process_image(image_path, output_dir, dataset)
    print(f"Processed image with {len(binary_masks)} masks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2 segmentation on an image.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()
    run_segmentation(dataset=args.dataset, image_path=args.image)
