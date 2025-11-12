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
from dinov3_encoder import DinoV3FeatureExtractor

# Global SAM instance to avoid reloading for each object
sam = None
predictor = None
dinov3_extractor = None


def _resize_mask_for_features(mask_tensor, spatial_hw):
    """Resize the reference mask to match a feature map spatial size."""
    resized = F.interpolate(
        mask_tensor, size=spatial_hw, mode="bilinear", align_corners=False
    )
    resized = resized.squeeze()
    if resized.dim() == 3:
        resized = resized[0]
    return resized


def _aggregate_target_vector(feat_map, mask, method="mean_max"):
    """Aggregate masked features into a single target vector."""
    if mask.numel() == 0:
        return None
    mask_bool = mask > 0
    mask_pixels = feat_map[mask_bool]
    if mask_pixels.shape[0] == 0:
        return None

    if method == "mean":
        return mask_pixels.mean(0)
    if method == "max":
        return torch.max(mask_pixels, dim=0)[0]
    if method == "mean_max":
        feat_mean = mask_pixels.mean(0)
        feat_max = torch.max(mask_pixels, dim=0)[0]
        return 0.5 * feat_mean + 0.5 * feat_max

    return mask_pixels.mean(0)


def _normalize_vector(vec, eps=1e-6):
    """Normalize the last dimension of a tensor."""
    denom = vec.norm(dim=-1, keepdim=True).clamp(min=eps)
    return vec / denom


def _compute_similarity(target_feat, feat_map):
    """Compute cosine similarity map between a target vector and dense features."""
    c, h, w = feat_map.shape
    feat_norm = feat_map / (feat_map.norm(dim=0, keepdim=True) + 1e-8)
    sim = target_feat @ feat_norm.reshape(c, h * w)
    return sim.reshape(1, 1, h, w)


def _prepare_mask_from_image(mask_image, size_hw, device):
    """
    Convert the raw RGB mask image into a float tensor aligned with a target resolution.
    """
    if mask_image.ndim == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask_image
    mask_tensor = torch.from_numpy(mask_gray.astype(np.float32) / 255.0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(mask_tensor, size=size_hw, mode="bilinear", align_corners=False)
    mask_tensor = mask_tensor.squeeze()
    return mask_tensor


def _smooth_mask(mask_array, method='none', kernel=5, sigma=1.0):
    """Apply simple post-processing to smooth binary masks."""
    if method == 'none':
        return mask_array.astype(bool)
    kernel = max(1, kernel)
    if kernel % 2 == 0:
        kernel += 1
    mask_float = mask_array.astype(np.float32)
    if method == 'gaussian':
        smoothed = cv2.GaussianBlur(mask_float, (kernel, kernel), sigma)
        return smoothed > 0.5
    return mask_array.astype(bool)


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
    parser.add_argument('--feature_encoder', type=str, default='sam',
                        choices=['sam', 'dinov3'],
                        help='Backbone to compute similarity features.')
    parser.add_argument('--dinov3_model_name', type=str, default=None,
                        help='timm model name for DinoV3 encoder (required when feature_encoder=dinov3).')
    parser.add_argument('--dinov3_image_size', type=int, default=518,
                        help='Input size for DinoV3 image preprocessing.')
    parser.add_argument('--dinov3_output_size', type=int, default=64,
                        help='Spatial size to which DinoV3 features are upsampled (matches SAM by default).')
    parser.add_argument('--dinov3_precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Computation precision for DinoV3 backbone.')
    parser.add_argument('--dinov3_no_pretrained', action='store_true',
                        help='Disable loading pretrained weights for DinoV3 model.')
    parser.add_argument('--dinov3_sim_weight', type=float, default=0.5,
                        help='Blend weight for DinoV3 similarity map (0 uses SAM only, 1 uses DinoV3 only).')
    parser.add_argument('--dinov3_sim_gain', type=float, default=1.0,
                        help='Optional scaling applied to the blended similarity map before downstream steps.')
    parser.add_argument('--mask_smoothing', type=str, default='none',
                        choices=['none', 'gaussian'],
                        help='Post-processing to smooth the final binary mask for cleaner edges.')
    parser.add_argument('--mask_smoothing_kernel', type=int, default=5,
                        help='Kernel size (odd integer) for mask smoothing filters.')
    parser.add_argument('--mask_smoothing_sigma', type=float, default=1.0,
                        help='Sigma used by gaussian smoothing (if enabled).')
    
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
    global sam, predictor, dinov3_extractor
    print("======> Load SAM (Global Instance)")
    if sam is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.sam_type == 'vit_h':
            sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        elif args.sam_type == 'vit_t':
            sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
            sam.eval()
        else:
            raise ValueError(f"Unknown SAM type: {args.sam_type}")
        sam.eval()
        predictor = SamPredictor(sam)
        print(f"======> SAM loaded on device: {device}")

        if args.feature_encoder == 'dinov3':
            if not args.dinov3_model_name:
                raise ValueError("Please provide --dinov3_model_name when using DinoV3 encoder.")
            if not (0.0 <= args.dinov3_sim_weight <= 1.0):
                raise ValueError("--dinov3_sim_weight must be in [0, 1].")
            precision = torch.float16 if args.dinov3_precision == 'fp16' else torch.float32
            dinov3_extractor = DinoV3FeatureExtractor(
                model_name=args.dinov3_model_name,
                image_size=args.dinov3_image_size,
                output_size=args.dinov3_output_size,
                device=torch.device(device),
                precision=precision,
                pretrained=not args.dinov3_no_pretrained,
            )
            print(f"======> DinoV3 encoder loaded: {args.dinov3_model_name}")
    
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
    feature_encoder = args.feature_encoder

    print("======> Obtain Location Prior" )
    # Image features encoding
    ref_mask_tensor = predictor.set_image(ref_image, ref_mask)  # stored for resizing

    sam_feat_map = predictor.features.squeeze().permute(1, 2, 0)
    sam_mask = _resize_mask_for_features(ref_mask_tensor, sam_feat_map.shape[:2])
    sam_target_vec = _aggregate_target_vector(sam_feat_map, sam_mask, args.feat_aggregation)
    if sam_target_vec is None:
        print(f"Warning: Reference mask is empty for {obj_name} (SAM features).")
        return
    sam_target_unit = _normalize_vector(sam_target_vec.unsqueeze(0))
    sam_target_embedding = sam_target_vec.unsqueeze(0).unsqueeze(0)
    dino_target_unit = None
    if feature_encoder == 'dinov3':
        dino_feat_map = dinov3_extractor(ref_image).permute(1, 2, 0)
        dino_mask = _prepare_mask_from_image(
            ref_mask,
            (dino_feat_map.shape[0], dino_feat_map.shape[1]),
            dinov3_extractor.device,
        )
        dino_target_vec = _aggregate_target_vector(dino_feat_map, dino_mask, args.feat_aggregation)
        if dino_target_vec is None:
            print(f"Warning: Reference mask is empty for {obj_name} (DinoV3 features).")
            return
        dino_target_unit = _normalize_vector(dino_target_vec.unsqueeze(0))
    else:
        dino_target_unit = None


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
        sam_test_feat = predictor.features.squeeze()
        sam_sim = _compute_similarity(sam_target_unit, sam_test_feat)

        if feature_encoder == 'dinov3':
            dino_test_feat = dinov3_extractor(test_image)
            dino_sim = _compute_similarity(dino_target_unit, dino_test_feat)
            weight = max(0.0, min(1.0, args.dinov3_sim_weight))
            sim = weight * dino_sim + (1 - weight) * sam_sim
        else:
            sim = sam_sim
        sim = sim * args.dinov3_sim_gain
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
            target_embedding=sam_target_embedding  # Target-semantic Prompting
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
            refined_mask = _smooth_mask(
                masks[best_idx],
                method=args.mask_smoothing,
                kernel=args.mask_smoothing_kernel,
                sigma=args.mask_smoothing_sigma,
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(test_image)
            show_mask(refined_mask, plt.gca())
            show_points(topk_xy, topk_label, plt.gca())
            plt.title(f"Mask {best_idx} (Score: {scores[best_idx]:.3f})", fontsize=18)
            plt.axis('off')
            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
            plt.savefig(vis_mask_output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()

            # Save mask
            final_mask = refined_mask
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
