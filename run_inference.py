import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

# ------------------------------------------------------------------------------------------------
# EDIT THESE VARIABLES
# ------------------------------------------------------------------------------------------------

# Path to your input image
IMAGE_PATH = ".asset/house_interior.jpg" 

# Text prompt: what you want to detect (separate classes with dots)
TEXT_PROMPT = "perfume bottle"

# Thresholds to filter results
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Output file path
OUTPUT_PATH = "outputs/my_result.jpg"

# ------------------------------------------------------------------------------------------------
# SETUP AND INFERENCE
# ------------------------------------------------------------------------------------------------

# Model config and checkpoint (you shouldn't need to change these if setup is correct)
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

def main():
    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Loading model...")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    print(f"Loading image: {IMAGE_PATH}")
    image_source, image = load_image(IMAGE_PATH)

    print(f"Running inference with prompt: '{TEXT_PROMPT}'")
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device="cuda" # Force GPU
    )

    print(f"Found {len(boxes)} objects. Annotating image...")
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    
    cv2.imwrite(OUTPUT_PATH, annotated_frame)
    print(f"Done! Saved result to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
