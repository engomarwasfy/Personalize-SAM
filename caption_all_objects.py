import os
import csv
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import types

def apply_mask_and_crop(image_path, mask_path):
    # Load image and mask
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return None
        
    # Threshold mask at 0 to catch any non-zero pixels
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    
    # Apply mask (black out background)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Find bounding box
    coords = cv2.findNonZero(mask)
    if coords is None:
        print(f"No non-zero pixels found in mask for {mask_path}")
        return None
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop
    cropped_image = masked_image[y:y+h, x:x+w]
    
    return Image.fromarray(cropped_image)

def load_blip2_model(device):
    model_id = "microsoft/Florence-2-large"
    
    # Load Florence-2 model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

def generate_caption(model, processor, image, device):
    # Florence-2 task prompt for brief captioning
    prompt = "<CAPTION>"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Use use_cache=False to avoid past_key_values bug
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=20,
        do_sample=False,
        num_beams=1,
        use_cache=False  # Disable KV cache to avoid the bug
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Remove the task prompt if present
    caption = generated_text.replace("<CAPTION>", "").strip()
    
    return caption

def main():
    parser = argparse.ArgumentParser(description="Caption objects in data folder using Florence-2")
    parser.add_argument("--data", type=str, default="./data", help="Root data directory")
    parser.add_argument("--output", type=str, default="generated_captions.csv", help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    images_root = os.path.join(args.data, "Images")
    annotations_root = os.path.join(args.data, "Annotations")

    print(f"Loading Florence-2 model on {args.device}...")
    try:
        model, processor = load_blip2_model(args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []
    
    if not os.path.exists(images_root):
        print(f"Error: {images_root} does not exist.")
        return

    objects = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
    print(f"Found {len(objects)} objects.")

    for obj_name in tqdm(objects):
        img_dir = os.path.join(images_root, obj_name)
        ann_dir = os.path.join(annotations_root, obj_name)
        
        # Use the first frame
        img_path = os.path.join(img_dir, "00.jpg")
        mask_path = os.path.join(ann_dir, "00.png")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for {obj_name}")
            continue
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {obj_name}")
            continue
            
        try:
            # Prepare image
            pil_image = apply_mask_and_crop(img_path, mask_path)
            if pil_image is None:
                print(f"Warning: Could not process mask for {obj_name}")
                continue
                
            # Generate caption
            caption = generate_caption(model, processor, pil_image, args.device)
            results.append({"object_name": obj_name, "caption": caption})
            
        except Exception as e:
            print(f"Error processing {obj_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save to CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object_name", "caption"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved captions to {args.output}")

if __name__ == "__main__":
    main()
