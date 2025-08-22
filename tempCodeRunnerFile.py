def generate_gradcam_results(model, image_path, img_pre, pred_idx, pred_label, pred_prob, output_dir):
    from pathlib import Path
    import os
    from PIL import Image

    # Load original image
    orig = Image.open(image_path).convert("RGB")

    # Function to get last conv layer
    def find_last_conv_layer(model):
        return "conv5_block3_out"  # Correct last conv layer

    last_conv = find_last_conv_layer(model)

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_pre, model, last_conv, pred_index=pred_idx)
    overlay = overlay_heatmap_on_image_pil(orig, heatmap)

    # Save overlay and original images
    base_name = Path(image_path).stem
    overlay_path = os.path.join(output_dir, f'{base_name}_gradcam.png')
    orig_path = os.path.join(output_dir, f'{base_name}_orig.png')
    overlay.save(overlay_path)
    orig.save(orig_path)

    # Return results
    return {
        'image': image_path,
        'pred_label': pred_label,
        'pred_prob': pred_prob,
        'overlay_path': overlay_path,
        'orig_path': orig_path
    }

# =======================
# Example usage
# =======================
result = generate_gradcam_results(
    model=model,
    image_path="dataset/Testing/meningioma/Te-me_0020.jpg",
    img_pre=img_pre,      # preprocessed image
    pred_idx=pred_idx,    # predicted class index
    pred_label=pred_label,
    pred_prob=pred_prob,
    output_dir="outputs"
)

print(result)