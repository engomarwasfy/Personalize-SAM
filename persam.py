"""
Enhanced PerSAM - Training-Free Personalized Segment Anything Model

This is a training-free approach that enhances the original PerSAM with:
- Better feature aggregation (mean_max combination for improved target embedding)
- Global SAM instance for efficiency (no model reloading between objects)
- Enhanced error handling and edge case management
- Configurable top-k point selection
- Improved similarity normalization with stability improvements
- Better mask refinement with padded bounding boxes
- Memory optimization with periodic cache clearing
- Better interpolation with align_corners for consistency

All enhancements are inference-only - no training or learnable parameters.
"""

import numpy as np
import torch
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor


# Global SAM instance to avoid reloading for each object
sam = None
predictor = None


def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--topk', type=int, default=1, help='Number of top-k points for location prior')
    parser.add_argument('--feat_aggregation', type=str, default='mean_max', 
                        choices=['mean', 'max', 'mean_max'], 
                        help='Feature aggregation method: mean, max, or mean_max combination')
    parser.add_argument('--sim_threshold', type=float, default=None, 
                        help='Similarity threshold for filtering (None for auto)')
    parser.add_argument('--box_padding', type=float, default=0.05, 
                        help='Padding ratio for bounding box expansion (0.05 = 5%%)')
    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    # Initialize global SAM instance
    global sam, predictor
    print("======> Load SAM (Global Instance)")
    if sam is None:
        if args.sam_type == 'vit_h':
            sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        elif args.sam_type == 'vit_t':
            sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
            sam.eval()
        else:
            raise ValueError(f"Unknown SAM type: {args.sam_type}")
        sam.eval()
        predictor = SamPredictor(sam)
        print(f"======> SAM loaded on device: {device}")
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            try:
                persam(args, obj_name, images_path, masks_path, output_path)
            except Exception as e:
                print(f"Error processing {obj_name}: {str(e)}")
                continue


def persam(args, obj_name, images_path, masks_path, output_path):

    print("\n------------> Segment " + obj_name)
    
    # Path preparation
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    # Check if files exist
    if not os.path.exists(ref_image_path):
        print(f"Warning: Reference image not found: {ref_image_path}")
        return
    if not os.path.exists(ref_mask_path):
        print(f"Warning: Reference mask not found: {ref_mask_path}")
        return
    if not os.path.exists(test_images_path):
        print(f"Warning: Test images path not found: {test_images_path}")
        return

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        print(f"Error: Failed to load reference image: {ref_image_path}")
        return
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    if ref_mask is None:
        print(f"Error: Failed to load reference mask: {ref_mask_path}")
        return
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    # Use global predictor
    global predictor
    if predictor is None:
        raise RuntimeError("Predictor not initialized. Call main() first.")

    print("======> Obtain Location Prior" )
    # Image features encoding
    ref_mask_tensor = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    # Better interpolation with align_corners for consistent results
    ref_mask_tensor = F.interpolate(ref_mask_tensor, size=ref_feat.shape[0: 2], 
                                     mode="bilinear", align_corners=False)
    ref_mask_tensor = ref_mask_tensor.squeeze()[0]

    # Check if mask has valid pixels
    mask_pixels = ref_feat[ref_mask_tensor > 0]
    if mask_pixels.shape[0] == 0:
        print(f"Warning: Reference mask is empty for {obj_name}")
        return

    # Enhanced target feature extraction with configurable aggregation
    target_feat = ref_feat[ref_mask_tensor > 0]
    if args.feat_aggregation == 'mean':
        target_embedding = target_feat.mean(0).unsqueeze(0)
    elif args.feat_aggregation == 'max':
        target_embedding = torch.max(target_feat, dim=0)[0].unsqueeze(0)
    elif args.feat_aggregation == 'mean_max':
        # Combine mean and max for better feature representation
        target_feat_mean = target_feat.mean(0)
        target_feat_max = torch.max(target_feat, dim=0)[0]
        target_embedding = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
    else:
        target_embedding = target_feat.mean(0).unsqueeze(0)
    
    # Normalize target embedding
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)


    print('======> Start Testing')
    test_images = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg')])
    
    for test_file in tqdm(test_images):
        test_idx = test_file.replace('.jpg', '')
        test_image_path = os.path.join(test_images_path, test_file)
        
        # Load test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Warning: Failed to load test image: {test_image_path}")
            continue
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Enhanced cosine similarity computation
        C, h, w = test_feat.shape
        test_feat_norm = test_feat / (test_feat.norm(dim=0, keepdim=True) + 1e-8)  # Add epsilon for stability
        test_feat_flat = test_feat_norm.reshape(C, h * w)
        sim = target_feat @ test_feat_flat

        sim = sim.reshape(1, 1, h, w)
        # Better interpolation with align_corners
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear", align_corners=False)
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Apply similarity threshold if specified
        if args.sim_threshold is not None:
            sim = torch.clamp(sim, min=args.sim_threshold)

        # Enhanced point selection with configurable top-k
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=args.topk)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Enhanced similarity normalization for attention
        sim_mean = sim.mean()
        sim_std = torch.std(sim) + 1e-8  # Add epsilon for stability
        sim_normalized = (sim - sim_mean) / sim_std
        
        # Better interpolation for attention map
        sim_normalized = F.interpolate(sim_normalized.unsqueeze(0).unsqueeze(0), 
                                       size=(64, 64), mode="bilinear", align_corners=False)
        attn_sim = sim_normalized.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction with target guidance
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1: Use mask input from first prediction
        if logits is not None and logits.shape[0] > 0:
            masks, scores, logits, _ = predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        mask_input=logits[best_idx: best_idx + 1, :, :], 
                        multimask_output=True)
            best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2: Use bounding box + mask input
        if masks is not None and masks.shape[0] > 0 and np.any(masks[best_idx]):
            y, x = np.nonzero(masks[best_idx])
            if len(y) > 0 and len(x) > 0:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                
                # Add padding to bounding box for better context
                img_h, img_w = test_image.shape[:2]
                padding_x = int((x_max - x_min) * args.box_padding)
                padding_y = int((y_max - y_min) * args.box_padding)
                
                x_min = max(0, x_min - padding_x)
                x_max = min(img_w - 1, x_max + padding_x)
                y_min = max(0, y_min - padding_y)
                y_max = min(img_h - 1, y_max + padding_y)
                
                input_box = np.array([x_min, y_min, x_max, y_max])
                
                masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :] if logits is not None else None, 
                    multimask_output=True)
                best_idx = np.argmax(scores)

        # Save results
        if masks is not None and masks.shape[0] > 0:
            # Save visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(test_image)
            show_mask(masks[best_idx], plt.gca())
            show_points(topk_xy, topk_label, plt.gca())
            plt.title(f"Mask {best_idx} (Score: {scores[best_idx]:.3f})", fontsize=18)
            plt.axis('off')
            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
            plt.savefig(vis_mask_output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()

            # Save mask
            final_mask = masks[best_idx]
            mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_colors[final_mask, :] = np.array([[0, 0, 128]])
            mask_output_path = os.path.join(output_path, test_idx + '.png')
            cv2.imwrite(mask_output_path, mask_colors)
        else:
            print(f"Warning: No valid masks generated for {test_idx}")
        
        # Clear cache periodically to save memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def point_selection(mask_sim, topk=1):
    """
    Enhanced point selection with better handling of edge cases.
    Training-free inference method - no learnable parameters.
    
    Args:
        mask_sim: Similarity map tensor of shape (H, W)
        topk: Number of top-k points to select
        
    Returns:
        topk_xy: Positive point coordinates (topk, 2) in (x, y) format
        topk_label: Positive point labels (topk,)
        last_xy: Negative point coordinates (topk, 2) in (x, y) format
        last_label: Negative point labels (topk,)
    """
    # Ensure mask_sim is a tensor and on the same device
    if not isinstance(mask_sim, torch.Tensor):
        mask_sim = torch.tensor(mask_sim)
    
    # Get dimensions (height, width)
    w, h = mask_sim.shape  # Note: shape is (W, H) from postprocess_masks output
    
    # Flatten the similarity map
    mask_flat = mask_sim.flatten(0)
    
    # Adjust topk if necessary
    num_pixels = mask_flat.numel()
    if num_pixels < topk:
        topk = max(1, num_pixels)  # At least 1 point
    
    # Get top-k positive points (highest similarity) - matching original logic
    if topk > 0 and num_pixels > 0:
        topk_xy = mask_flat.topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()
    else:
        # Fallback to center point if no valid points
        topk_xy = np.array([[h // 2, w // 2]])
        topk_label = np.array([1])
        
    # Get top-k negative points (lowest similarity)
    if topk > 0 and num_pixels > topk:
        last_xy = mask_flat.topk(topk, largest=False)[1]
        last_x = (last_xy // h).unsqueeze(0)
        last_y = (last_xy - last_x * h)
        last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
        last_label = np.array([0] * topk)
        last_xy = last_xy.cpu().numpy()
    else:
        # Fallback to corner points if needed
        last_xy = np.array([[0, 0], [h-1, w-1]])[:topk]
        last_label = np.array([0] * last_xy.shape[0])
    
    return topk_xy, topk_label, last_xy, last_label
    

if __name__ == "__main__":
    main()
