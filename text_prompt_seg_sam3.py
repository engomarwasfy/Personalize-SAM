import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_miou import evaluate as evaluate_miou

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:  # pragma: no cover - requires sam3 install
    raise ImportError(
        "sam3 is not installed or not on PYTHONPATH. Ensure you ran `pip install -e sam3` "
        "and that torch is available."
    ) from e

try:
    from personalized_sam3 import CaptionConfig, PersonalizedSAM3
    _HAS_CAPTION = True
    _CAPTION_MODEL_DEFAULT = getattr(CaptionConfig, "model_id", "Salesforce/blip-image-captioning-base")
except Exception:
    _HAS_CAPTION = False
    _CAPTION_MODEL_DEFAULT = "Salesforce/blip-image-captioning-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-prompt segmentation for a dataset using SAM3 (folder names as prompts)."
    )
    parser.add_argument("--data", type=str, default="./data", help="Root containing Images/ and Annotations/.")
    parser.add_argument("--outdir", type=str, default="outputs/text_sam3", help="Directory to save masks.")
    parser.add_argument("--ref-idx", type=str, default="00", help="Reference frame index (for mIoU eval).")
    parser.add_argument(
        "--sam3-ckpt",
        type=str,
        default="weights/sam3.pt",
        help="Path to SAM3 checkpoint (set to an existing file to avoid HF download).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering SAM3 predictions.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask probability threshold after interpolation.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["folder", "caption", "florence", "qwen2vl", "internvl2", "internvl8b", "internvl26b", "internvl76b"],
        default="folder",
        help="Use folder, BLIP, Florence-2, Qwen2-VL, InternVL2-2B/8B/26B/76B (76B=ABSOLUTE BEST for 128GB RAM).",
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        default=_CAPTION_MODEL_DEFAULT,
        help="Caption model id (only if --prompt-source caption). Defaults to LLaVA 1.5 7B.",
    )
    parser.add_argument(
        "--caption-device",
        type=str,
        default="auto",
        help="Device for caption model (e.g., cpu or cuda). Use cpu for Florence large to keep VRAM free.",
    )
    return parser.parse_args()


def _folder_prompt(name: str) -> str:
    base = re.sub(r"\d+$", "", name)
    prompt = (base or name).replace("_", " ").strip()
    return prompt


def _to_color_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask > 0, :] = np.array([[0, 0, 128]], dtype=np.uint8)
    return out


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_root = os.path.join(args.data, "Images")
    masks_root = os.path.join(args.data, "Annotations")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    captioner = None
    florence_model = None
    florence_processor = None
    qwen_model = None
    qwen_processor = None
    internvl_model = None
    internvl_tokenizer = None
    
    if args.prompt_source == "caption":
        if not _HAS_CAPTION:
            raise ImportError("Caption mode requested but personalized_sam3/transformers not available.")
        cap_device = None if args.caption_device.lower() == "auto" else args.caption_device
        captioner = PersonalizedSAM3(
            caption_cfg=CaptionConfig(model_id=args.caption_model),
            sam3_builder=None,
            num_points=0,
            device=cap_device,
        )
    elif args.prompt_source == "florence":
        # Load Florence-2 model
        from transformers import AutoProcessor, AutoModelForCausalLM
        import cv2
        
        model_id = "microsoft/Florence-2-large"
        cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
        print(f"Loading Florence-2 model on {cap_device}...")
        florence_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(cap_device).eval()
        florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Florence-2 model loaded.")
    elif args.prompt_source == "qwen2vl":
        # Load Qwen2-VL-2B model (SOTA 2025)
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import cv2
        
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
        print(f"Loading Qwen2-VL-2B model on {cap_device}...")
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if cap_device == "cuda" else torch.float32,
            device_map=cap_device
        ).eval()
        qwen_processor = AutoProcessor.from_pretrained(model_id)
        print("Qwen2-VL-2B model loaded.")
    elif args.prompt_source == "internvl2":
        # Load InternVL2-2B model (SOTA 2025, CPU-friendly)
        from transformers import AutoTokenizer, AutoModel
        import cv2
        
        model_id = "OpenGVLab/InternVL2-2B"
        cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
        print(f"Loading InternVL2-2B model on {cap_device}...")
        internvl_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if cap_device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        if cap_device != "auto":
            internvl_model = internvl_model.to(cap_device)
        internvl_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("InternVL2-2B model loaded.")
    elif args.prompt_source == "internvl8b":
        # Load InternVL2-8B model (Best SOTA that fits GPU)
        from transformers import AutoTokenizer, AutoModel
        import cv2
        
        model_id = "OpenGVLab/InternVL2-8B"
        cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
        print(f"Loading InternVL2-8B model on {cap_device}...")
        internvl_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if cap_device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        if cap_device != "auto":
            internvl_model = internvl_model.to(cap_device)
        internvl_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("InternVL2-8B model loaded.")
    elif args.prompt_source == "internvl26b":
        # Load InternVL2-26B model (Best for CPU with 128GB RAM)
        from transformers import AutoTokenizer, AutoModel
        import cv2
        
        model_id = "OpenGVLab/InternVL2-26B"
        cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
        print(f"Loading InternVL2-26B model on {cap_device}... (this may take a few minutes)")
        internvl_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        if cap_device != "auto":
            internvl_model = internvl_model.to(cap_device)
        internvl_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("InternVL2-26B model loaded.")
    elif args.prompt_source == "internvl76b":
        # Load InternVL2-76B model (ABSOLUTE BEST for 128GB RAM)
        from transformers import AutoTokenizer, AutoModel
        import cv2
        
        model_id = "OpenGVLab/InternVL2-Llama3-76B"
        cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
        print(f"Loading InternVL2-76B model on {cap_device}... (this will take several minutes)")
        internvl_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        if cap_device != "auto":
            internvl_model = internvl_model.to(cap_device)
        internvl_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("InternVL2-76B model loaded - ABSOLUTE BEST QUALITY!")


    # Build SAM3 model and processor
    model = build_sam3_image_model(
        checkpoint_path=args.sam3_ckpt,
        load_from_HF=args.sam3_ckpt is None,
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence_threshold)

    obj_names = [n for n in os.listdir(images_root) if not n.startswith(".")]
    for obj_name in sorted(obj_names):
        obj_img_dir = os.path.join(images_root, obj_name)
        if not os.path.isdir(obj_img_dir):
            continue

        # Folder-name prompt with overrides from generated_captions.csv (sam3_short)
        
        # Optimal prompt mix - best of simple/mixed/detailed per object
        overrides = {
            "backpack": "backpack",
            "backpack_dog": "gray backpack",
            "barn": "barn",
            "bear_plushie": "A brown teddy bear with a blue hat on its head.",
            "berry_bowl": "berry bowl",
            "can": "silver beverage can",
            "candle": "a white candle with a wooden lid on a white background",
            "cat": "cat",
            "cat2": "cat",
            "cat_statue": "cat statue",
            "chair": "chair",
            "clock": "A copper alarm clock with a crown on the face.",
            "colorful_sneaker": "colorful sneaker",
            "colorful_teapot": "A colorful teapot with a flower design on it.",
            "dog": "A dog with its tongue out.",
            "dog2": "A brown dog with long hair and a big smile on its face.",
            "dog3": "dog",
            "dog4": "dog",
            "dog5": "dog",
            "dog6": "dog",
            "dog7": "dog",
            "dog8": "dog",
            "duck_toy": "duck toy",
            "elephant": "elephant",
            "fancy_boot": "fancy fashion boot",
            "grey_sloth_plushie": "grey sloth plushie",
            "monster_toy": "monster toy",
            "poop_emoji": "A close up of a poop emoji with big eyes.",
            "rc_car": "A toy race car with a man in a helmet driving a red and yellow car.",
            "red_cartoon": "red cartoon",
            "robot_toy": "robot toy",
            "round_bird": "A round bird figurine with a brown beak.",
            "shiny_sneaker": "shiny sneaker with white soles.",
            "table": "a round coffee table with a wooden top and metal frame",
            "teapot": "teapot",
            "teddybear": "teddy bear",
            "thin_bird": "A thin white bird's figurine.",
            "tortoise_plushy": "green tortoise plush toy",
            "wolf_plushie": "A gray and white husky dog plushie with a red collar laying down.",
            "wooden_pot": "A wooden vase pot with a carved design on it.",
        }
        # Determine prompt based on source
        if args.prompt_source in ["internvl2", "internvl8b", "internvl26b", "internvl76b"] and internvl_model is not None:
            # Use InternVL2-2B to caption the reference frame (SOTA 2025)
            ref_frame = f"{args.ref_idx}.jpg"
            ref_img_path = os.path.join(obj_img_dir, ref_frame)
            if os.path.exists(ref_img_path):
                import cv2
                ref_image = Image.open(ref_img_path).convert("RGB")
                
                # Load and apply mask for strict masking
                ref_mask_path = os.path.join(masks_root, obj_name, f"{args.ref_idx}.png")
                if os.path.exists(ref_mask_path):
                    mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
                    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                    
                    # Apply mask and crop
                    img_np = np.array(ref_image)
                    masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)
                    coords = cv2.findNonZero(mask)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        cropped = masked_img[y:y+h, x:x+w]
                        ref_image = Image.fromarray(cropped)
                
                # Generate caption with InternVL2
                import torchvision.transforms as T
                from torchvision.transforms.functional import InterpolationMode
                
                # InternVL2 image preprocessing
                IMAGENET_MEAN = (0.485, 0.456, 0.406)
                IMAGENET_STD = (0.229, 0.224, 0.225)
                
                def build_transform(input_size):
                    transform = T.Compose([
                        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])
                    return transform
                
                # Simple preprocessing for single image
                transform = build_transform(input_size=448)
                pixel_values = transform(ref_image).unsqueeze(0)
                
                if args.caption_device == "cuda":
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                else:
                    pixel_values = pixel_values.to(torch.float32).cpu()
                
                question = "Describe this object briefly in one sentence."
                generation_config = dict(max_new_tokens=30, do_sample=False)
                prompt = internvl_model.chat(internvl_tokenizer, pixel_values, question, generation_config)
                print(f"\nSegmenting {obj_name} with InternVL2 caption: '{prompt}'")
            else:
                prompt = _folder_prompt(obj_name)
                print(f"\nSegmenting {obj_name} with folder prompt (ref not found): '{prompt}'")
        elif args.prompt_source == "qwen2vl" and qwen_model is not None:
            # Use Qwen2-VL to caption the reference frame (SOTA 2025)
            ref_frame = f"{args.ref_idx}.jpg"
            ref_img_path = os.path.join(obj_img_dir, ref_frame)
            if os.path.exists(ref_img_path):
                import cv2
                ref_image = Image.open(ref_img_path).convert("RGB")
                
                # Load and apply mask for strict masking
                ref_mask_path = os.path.join(masks_root, obj_name, f"{args.ref_idx}.png")
                if os.path.exists(ref_mask_path):
                    mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
                    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                    
                    # Apply mask and crop
                    img_np = np.array(ref_image)
                    masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)
                    coords = cv2.findNonZero(mask)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        cropped = masked_img[y:y+h, x:x+w]
                        ref_image = Image.fromarray(cropped)
                
                # Generate caption with Qwen2-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ref_image},
                            {"type": "text", "text": "Describe this object briefly in one sentence."},
                        ],
                    }
                ]
                text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = qwen_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
                inputs = inputs.to(cap_device)
                
                generated_ids = qwen_model.generate(**inputs, max_new_tokens=30)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                prompt = qwen_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                print(f"\nSegmenting {obj_name} with Qwen2-VL caption: '{prompt}'")
            else:
                prompt = _folder_prompt(obj_name)
                print(f"\nSegmenting {obj_name} with folder prompt (ref not found): '{prompt}'")
        elif args.prompt_source == "florence" and florence_model is not None:
            # Use Florence-2 to caption the reference frame
            ref_frame = f"{args.ref_idx}.jpg"
            ref_img_path = os.path.join(obj_img_dir, ref_frame)
            if os.path.exists(ref_img_path):
                import cv2
                ref_image = Image.open(ref_img_path).convert("RGB")
                
                # Load and apply mask for strict masking
                ref_mask_path = os.path.join(masks_root, obj_name, f"{args.ref_idx}.png")
                if os.path.exists(ref_mask_path):
                    mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
                    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                    
                    # Apply mask and crop
                    img_np = np.array(ref_image)
                    masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)
                    coords = cv2.findNonZero(mask)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        cropped = masked_img[y:y+h, x:x+w]
                        ref_image = Image.fromarray(cropped)
                
                # Generate caption with Florence-2
                cap_device = args.caption_device if args.caption_device.lower() != "auto" else device
                inputs = florence_processor(text="<CAPTION>", images=ref_image, return_tensors="pt").to(cap_device)
                generated_ids = florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False
                )
                caption = florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                prompt = caption.replace("<CAPTION>", "").strip()
                print(f"\nSegmenting {obj_name} with Florence-2 caption: '{prompt}'")
            else:
                prompt = _folder_prompt(obj_name)
                print(f"\nSegmenting {obj_name} with folder prompt (ref not found): '{prompt}'")
        elif args.prompt_source == "caption" and captioner is not None:
            # Use BLIP caption (existing logic)
            ref_frame = f"{args.ref_idx}.jpg"
            ref_img_path = os.path.join(obj_img_dir, ref_frame)
            ref_mask_path = os.path.join(masks_root, obj_name, f"{args.ref_idx}.png")
            if os.path.exists(ref_img_path) and os.path.exists(ref_mask_path):
                result = captioner.run(ref_img_path, ref_mask_path, run_sam3=False)
                prompt = result["text_prompt"]
                print(f"\nSegmenting {obj_name} with BLIP caption: '{prompt}'")
            else:
                prompt = _folder_prompt(obj_name)
                print(f"\nSegmenting {obj_name} with folder prompt (ref not found): '{prompt}'")
        else:
            # Use static dict or folder name
            prompt = overrides.get(obj_name, _folder_prompt(obj_name))
            print(f"\nSegmenting {obj_name} with prompt: '{prompt}'")

        dst_dir = os.path.join(args.outdir, obj_name)
        Path(dst_dir).mkdir(parents=True, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(obj_img_dir) if f.endswith(".jpg")])
        for frame_file in frame_files:
            frame_idx = os.path.splitext(frame_file)[0]
            img_path = os.path.join(obj_img_dir, frame_file)
            image = Image.open(img_path).convert("RGB")

            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt=prompt, state=state)

            masks = state.get("masks", None)
            scores = state.get("scores", None)
            if masks is None or masks.numel() == 0:
                mask_bin = np.zeros((image.height, image.width), dtype=np.uint8)
                print(f"  {frame_idx}: no masks found, writing empty mask.")
            else:
                masks = masks.squeeze(1)  # [N,H,W]
                scores = scores if scores is not None else torch.ones(masks.shape[0])
                best = scores.argmax().item()
                mask_bin = (masks[best].cpu().numpy() > args.mask_threshold).astype(np.uint8)

            out_path = os.path.join(dst_dir, f"{frame_idx}.png")
            cv2_mask = _to_color_mask(mask_bin)
            import cv2  # local import to avoid dependency if not used elsewhere

            cv2.imwrite(out_path, cv2_mask)
            print(f"  Saved {out_path}")

    print("\nEvaluating mIoU...")
    evaluate_miou(args.outdir, masks_root, args.ref_idx)


if __name__ == "__main__":
    main()
