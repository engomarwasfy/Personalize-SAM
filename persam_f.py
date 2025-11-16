import numpy as np
import torch
import torch.nn as nn
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


def _resize_mask_for_features(mask_tensor, spatial_hw):
    """Resize mask tensor to match feature map resolution."""
    resized = F.interpolate(mask_tensor, size=spatial_hw, mode="bilinear", align_corners=False)
    resized = resized.squeeze()
    if resized.dim() == 3:
        resized = resized[0]
    return resized


def _aggregate_target_vector(feat_map, mask, method="mean_max"):
    """Aggregate masked features into a single vector."""
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
    """L2-normalize along last dimension."""
    denom = vec.norm(dim=-1, keepdim=True).clamp(min=eps)
    return vec / denom


def _compute_similarity(target_feat, feat_map):
    """Compute cosine similarity map."""
    c, h, w = feat_map.shape
    feat_norm = feat_map / (feat_map.norm(dim=0, keepdim=True) + 1e-8)
    sim = target_feat @ feat_norm.reshape(c, h * w)
    return sim.reshape(1, 1, h, w)


def _prepare_mask_from_image(mask_image, size_hw, device):
    """Resize raw RGB mask into tensor aligned with target size."""
    if mask_image.ndim == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask_image
    mask_tensor = torch.from_numpy(mask_gray.astype(np.float32) / 255.0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(mask_tensor, size=size_hw, mode="bilinear", align_corners=False)
    mask_tensor = mask_tensor.squeeze()
    return mask_tensor


def _smooth_mask(mask_array, method='none', kernel_size=5, sigma=1.0):
    """Apply optional smoothing to a binary mask."""
    if method == 'none':
        return mask_array
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    mask_float = mask_array.astype(np.float32)
    if method == 'gaussian':
        smoothed = cv2.GaussianBlur(mask_float, (k, k), sigma)
        return smoothed > 0.5
    return mask_array

def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam_f')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')

    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--log_epoch', type=int, default=200)
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--feat_aggregation', type=str, default='mean_max',
                        choices=['mean', 'max', 'mean_max'],
                        help='Feature aggregation strategy for target vector.')
    parser.add_argument('--feature_encoder', type=str, default='sam',
                        choices=['sam', 'dinov3'],
                        help='Backbone to compute similarity features.')
    parser.add_argument('--dinov3_model_name', type=str, default=None,
                        help='timm model name for DinoV3 encoder (required when feature_encoder=dinov3).')
    parser.add_argument('--dinov3_image_size', type=int, default=518,
                        help='Input resolution for DinoV3 preprocessing.')
    parser.add_argument('--dinov3_output_size', type=int, default=64,
                        help='Spatial size to which DinoV3 features are upsampled.')
    parser.add_argument('--dinov3_precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Precision for DinoV3 backbone.')
    parser.add_argument('--dinov3_no_pretrained', action='store_true',
                        help='Disable pretrained weights for DinoV3 model.')
    parser.add_argument('--dinov3_sim_weight', type=float, default=0.5,
                        help='Blend weight for DinoV3 similarity map (0 = SAM only, 1 = Dino only).')
    parser.add_argument('--dinov3_sim_gain', type=float, default=1.0,
                        help='Scaling applied to blended similarity map.')
    parser.add_argument('--dinov3_train_weight', type=float, default=None,
                        help='Blend weight for DinoV3 similarity during training (defaults to dinov3_sim_weight).')
    parser.add_argument('--topk', type=int, default=1,
                        help='Number of top-k points for location prior.')
    parser.add_argument('--sim_threshold', type=float, default=None,
                        help='Similarity threshold for filtering (None for auto).')
    parser.add_argument('--box_padding', type=float, default=0.05,
                        help='Padding ratio for bounding box expansion.')
    parser.add_argument('--mask_smoothing', type=str, default='none',
                        choices=['none', 'gaussian'],
                        help='Optional smoothing to apply on the final binary mask.')
    parser.add_argument('--mask_smoothing_kernel', type=int, default=5,
                        help='Kernel size for mask smoothing filters (odd integer).')
    parser.add_argument('--mask_smoothing_sigma', type=float, default=1.0,
                        help='Sigma parameter for Gaussian mask smoothing.')
    parser.add_argument('--use_negative_points', action='store_true',
                        help='Include negative (background) points when selecting prompts.')
    
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
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            persam_f(args, obj_name, images_path, masks_path, output_path)


def persam_f(args, obj_name, images_path, masks_path, output_path):
    
    print("\n------------> Segment " + obj_name)
    
    # Path preparation
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).to(device)

    
    print("======> Load SAM" )
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    
    
    for name, param in sam.named_parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)
    dino_extractor = None
    use_dino = args.feature_encoder == 'dinov3'
    if use_dino:
        if not args.dinov3_model_name:
            raise ValueError("Please provide --dinov3_model_name when using DinoV3 encoder.")
        if not (0.0 <= args.dinov3_sim_weight <= 1.0):
            raise ValueError("--dinov3_sim_weight must be in [0, 1].")
        if args.dinov3_train_weight is not None and not (0.0 <= args.dinov3_train_weight <= 1.0):
            raise ValueError("--dinov3_train_weight must be in [0, 1].")
        precision = torch.float16 if args.dinov3_precision == 'fp16' else torch.float32
        dino_extractor = DinoV3FeatureExtractor(
            model_name=args.dinov3_model_name,
            image_size=args.dinov3_image_size,
            output_size=args.dinov3_output_size,
            device=device,
            precision=precision,
            pretrained=not args.dinov3_no_pretrained,
        )
        print(f"======> DinoV3 encoder loaded: {args.dinov3_model_name}")
    

    print("======> Obtain Self Location Prior" )
    ref_mask_tensor = predictor.set_image(ref_image, ref_mask)
    sam_feat_map = predictor.features.squeeze().permute(1, 2, 0)
    sam_mask = _resize_mask_for_features(ref_mask_tensor, sam_feat_map.shape[:2])
    sam_target_vec = _aggregate_target_vector(sam_feat_map, sam_mask, args.feat_aggregation)
    if sam_target_vec is None:
        print(f"Warning: Reference mask is empty for {obj_name} (SAM features).")
        return
    target_feat = _normalize_vector(sam_target_vec.unsqueeze(0))
    sam_features = predictor.features.squeeze()
    C, h, w = sam_features.shape
    sam_features_norm = sam_features / (sam_features.norm(dim=0, keepdim=True) + 1e-8)
    sam_sim = target_feat @ sam_features_norm.reshape(C, h * w)
    sam_sim = sam_sim.reshape(1, 1, h, w)

    dino_target_unit = None
    if use_dino:
        dino_ref_feat = dino_extractor(ref_image)
        dino_mask = _prepare_mask_from_image(
            ref_mask,
            (dino_ref_feat.shape[1], dino_ref_feat.shape[2]),
            dino_extractor.device,
        )
        dino_target_vec = _aggregate_target_vector(
            dino_ref_feat.permute(1, 2, 0), dino_mask, args.feat_aggregation
        )
        if dino_target_vec is None:
            print(f"Warning: Reference mask is empty for {obj_name} (DinoV3 features).")
            return
        dino_target_unit = _normalize_vector(dino_target_vec.unsqueeze(0))
        dino_sim = _compute_similarity(dino_target_unit, dino_ref_feat)
        train_weight = args.dinov3_train_weight
        if train_weight is None:
            train_weight = args.dinov3_sim_weight
        sim = train_weight * dino_sim + (1 - train_weight) * sam_sim
    else:
        sim = sam_sim

    sim = F.interpolate(sim, scale_factor=4, mode="bilinear", align_corners=False)
    sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()

    if args.sim_threshold is not None:
        sim = torch.clamp(sim, min=args.sim_threshold)

    # Positive/negative location prior
    pos_xy, pos_label, neg_xy, neg_label = point_selection(
        sim, topk=args.topk, include_negative=args.use_negative_points
    )
    if neg_xy.size > 0:
        topk_xy = np.concatenate([pos_xy, neg_xy], axis=0)
        topk_label = np.concatenate([pos_label, neg_label], axis=0)
    else:
        topk_xy, topk_label = pos_xy, pos_label


    print('======> Start Training')
    # Learnable mask weights
    mask_weights = Mask_Weights().to(device)
    mask_weights.train()
    
    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

    for train_idx in range(args.train_epoch):

        # Run the decoder
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        logits_high = logits_high.flatten(1)

        # Weighted sum three-scale masks
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        logits_high = logits_high * weights
        logits_high = logits_high.sum(0).unsqueeze(0)

        dice_loss = calculate_dice_loss(logits_high, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
        loss = dice_loss + focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if train_idx % args.log_epoch == 0:
            print('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))


    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    weights_np = weights.detach().cpu().numpy()
    print('======> Mask weights:\n', weights_np)

    print('======> Start Testing')
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):

        # Load test image
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Cosine similarity with optional DinoV3 blending
        C, h, w = test_feat.shape
        test_feat_norm = test_feat / (test_feat.norm(dim=0, keepdim=True) + 1e-8)
        sam_sim = target_feat @ test_feat_norm.reshape(C, h * w)
        sam_sim = sam_sim.reshape(1, 1, h, w)
        if use_dino:
            dino_test_feat = dino_extractor(test_image)
            dino_sim = _compute_similarity(dino_target_unit, dino_test_feat)
            sim = args.dinov3_sim_weight * dino_sim + (1 - args.dinov3_sim_weight) * sam_sim
        else:
            sim = sam_sim

        sim = sim * args.dinov3_sim_gain

        sim = F.interpolate(sim, scale_factor=4, mode="bilinear", align_corners=False)
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()
        if args.sim_threshold is not None:
            sim = torch.clamp(sim, min=args.sim_threshold)

        # Positive/negative location prior
        pos_xy, pos_label, neg_xy, neg_label = point_selection(
            sim, topk=args.topk, include_negative=args.use_negative_points
        )
        if neg_xy.size > 0:
            topk_xy = np.concatenate([pos_xy, neg_xy], axis=0)
            topk_label = np.concatenate([pos_label, neg_label], axis=0)
        else:
            topk_xy, topk_label = pos_xy, pos_label

        # First-step prediction
        masks, scores, logits, logits_high = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)

        # Weighted sum three-scale masks
        logits_high = logits_high * weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        logits = logits * weights_np[..., None]
        logit = logits.sum(0)

        # Cascaded Post-refinement-1
        y, x = np.nonzero(mask)
        if len(y) == 0 or len(x) == 0:
            continue
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
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
            mask_input=logit[None, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        if len(y) == 0 or len(x) == 0:
            continue
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
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
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)
        
        # Save masks
        refined_mask = _smooth_mask(
            masks[best_idx],
            method=args.mask_smoothing,
            kernel_size=args.mask_smoothing_kernel,
            sigma=args.mask_smoothing_sigma,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(refined_mask, plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title(f"Mask {best_idx}", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')

        final_mask = refined_mask
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_idx + '.png')
        cv2.imwrite(mask_output_path, mask_colors)


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def point_selection(mask_sim, topk=1, include_negative=False):
    """
    Select top-k positive (and optionally negative) points from similarity map.
    Returns positive coords/labels and negative coords/labels.
    """
    if not isinstance(mask_sim, torch.Tensor):
        mask_sim = torch.tensor(mask_sim)

    w, h = mask_sim.shape
    mask_flat = mask_sim.flatten(0)
    num_pixels = mask_flat.numel()
    if num_pixels < topk:
        topk = max(1, num_pixels)

    if topk > 0 and num_pixels > 0:
        pos_idx = mask_flat.topk(topk)[1]
        pos_x = (pos_idx // h).unsqueeze(0)
        pos_y = (pos_idx - pos_x * h)
        pos_xy = torch.cat((pos_y, pos_x), dim=0).permute(1, 0)
        pos_xy = pos_xy.cpu().numpy()
        pos_label = np.array([1] * topk)
    else:
        pos_xy = np.array([[h // 2, w // 2]])
        pos_label = np.array([1])

    if include_negative and topk > 0 and num_pixels > topk:
        neg_idx = mask_flat.topk(topk, largest=False)[1]
        neg_x = (neg_idx // h).unsqueeze(0)
        neg_y = (neg_idx - neg_x * h)
        neg_xy = torch.cat((neg_y, neg_x), dim=0).permute(1, 0)
        neg_xy = neg_xy.cpu().numpy()
        neg_label = np.array([0] * topk)
    elif include_negative:
        neg_xy = np.array([[0, 0], [h - 1, w - 1]])[:topk]
        neg_label = np.array([0] * neg_xy.shape[0])
    else:
        neg_xy = np.empty((0, 2), dtype=np.int64)
        neg_label = np.empty((0,), dtype=np.int64)

    return pos_xy, pos_label, neg_xy, neg_label


def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


if __name__ == '__main__':
    main()
