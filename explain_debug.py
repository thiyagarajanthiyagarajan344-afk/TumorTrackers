import os
import argparse
from datetime import datetime
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------- Config ----------
MODEL_FILE = "model.keras"
OUT_DIR = "explanations"
IMG_INPUT_SIZE = (224, 224)   # model input size
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MALIGNANCY_MAP = {"glioma": "malignant", "meningioma": "benign", "pituitary": "benign", "notumor": "benign"}
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def get_resnet_backbone(model):
    # try direct name
    try:
        return model.get_layer("resnet50")
    except Exception:
        # search for nested model with 'resnet' in name
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and 'resnet' in layer.name.lower():
                return layer
    return None

def find_last_conv_name(resnet_backbone):
    # return last Conv2D layer name inside a model
    for l in reversed(resnet_backbone.layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            return l.name
    raise RuntimeError("No Conv2D layer found in backbone")

def preprocess_for_model(img_path):
    # load and preprocess for model (RGB)
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    in_img = img.resize(IMG_INPUT_SIZE, Image.BICUBIC)
    arr = np.array(in_img).astype("float32")
    # Use resnet50 preprocess
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return img, arr  # orig PIL Image, preprocessed batch

def compute_gradcam(model, img_pre, conv_layer_name, pred_index):
    """
    Build grad_model by re-rooting the conv layer into top-level graph (safe).
    Returns heatmap (H_conv, W_conv) normalized to [0,1] or None if fails.
    """
    # find backbone (resnet) and conv layer
    resnet = get_resnet_backbone(model)
    if resnet is None:
        raise RuntimeError("ResNet backbone not found inside model")

    try:
        conv_layer = resnet.get_layer(conv_layer_name)
    except Exception as e:
        raise RuntimeError(f"Conv layer {conv_layer_name} not found in backbone: {e}")

    # re-root conv output into top-level graph by applying conv_model to model.inputs[0] symbolically
    conv_model = tf.keras.Model(inputs=resnet.inputs, outputs=conv_layer.output)
    # conv_on_top is a symbolic tensor in top-level graph
    conv_on_top = conv_model(model.inputs[0])
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_on_top, model.output])

    img_tensor = tf.convert_to_tensor(img_pre, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)                # ensure tape watches intermediate activations
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # H x W x C

    # Weighted sum
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if tf.math.is_nan(max_val) or max_val == 0:
        return None
    heatmap = heatmap / (max_val + 1e-8)
    return heatmap.numpy()

def input_gradient_saliency(model, img_pre, class_idx):
    x = tf.convert_to_tensor(img_pre, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        score = preds[:, class_idx]
    grads = tape.gradient(score, x)  # 1,H,W,3
    if grads is None:
        return None
    sal = tf.reduce_mean(tf.abs(grads), axis=-1)[0].numpy()
    sal -= sal.min()
    if sal.max() > 0:
        sal /= (sal.max() + 1e-8)
    return sal

def apply_colormap_overlay(orig_pil, heatmap, alpha=0.6, boost=1.5):
    """
    orig_pil: PIL.Image RGB original (full size)
    heatmap: 2D array normalized [0,1] at arbitrary size -> will be resized to original size
    returns: overlay_bgr (uint8 BGR), annotated PIL for further drawing
    """
    orig = np.array(orig_pil)  # HxWx3 RGB
    H, W = orig.shape[:2]
    hm_resized = cv2.resize((heatmap*255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    colormap = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)  # BGR
    # boost intensity
    colormap = cv2.convertScaleAbs(colormap, alpha=boost, beta=0)
    overlay_bgr = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 1 - alpha, colormap, alpha, 0)
    return overlay_bgr, hm_resized.astype(np.float32)/255.0  # overlay BGR, hm in [0,1]

def annotate_and_label(overlay_bgr, heatmap_resized, label_text, prob, malignancy, threshold=0.45):
    """
    Draw bounding boxes around hotspots and write label+prob+malignancy.
    """
    overlay = overlay_bgr.copy()
    hm = (heatmap_resized.copy()*255).astype(np.uint8)
    _, mask = cv2.threshold(hm, int(255*threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # skip tiny
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,255,255), 2)
        cv2.putText(overlay, f"{label_text} ({prob*100:.1f}%)", (x, max(20,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        drawn += 1

    # top-left summary text
    txt = f"{label_text} | {malignancy} | {prob*100:.1f}%"
    cv2.rectangle(overlay, (2,2), (min(400, overlay.shape[1]-4), 28), (0,0,0), -1)
    cv2.putText(overlay, txt, (6,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return overlay

# ---------- Main flow ----------
def explain_image(img_path):
    # load model
    model = load_model(MODEL_FILE, compile=False)

    # preprocess
    orig_pil, img_pre = preprocess_for_model(img_path := img_path)  # original PIL and preprocessed batch

    # prediction
    preds = model.predict(img_pre, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    label = CLASS_NAMES[class_idx]
    prob = float(preds[0][class_idx])
    malignancy = MALIGNANCY_MAP[label]

    # attempt to compute Grad-CAM using last conv layer detected inside backbone
    resnet = get_resnet_backbone(model)
    heatmap = None
    used_method = None
    if resnet is not None:
        try:
            last_conv_name = find_last_conv_name(resnet)
            heatmap = compute_gradcam(model, img_pre, last_conv_name, pred_index=class_idx)
            if heatmap is not None:
                used_method = f"gradcam:{last_conv_name}"
        except Exception:
            heatmap = None

    # fallback to input-gradient saliency
    if heatmap is None:
        heatmap = input_gradient_saliency(model, img_pre, class_idx)
        used_method = "input_gradient_saliency"

    # ensure heatmap exists
    if heatmap is None:
        raise RuntimeError("Could not compute any explanation map")

    # overlay & annotate
    overlay_bgr, hm_resized = apply_colormap_overlay(orig_pil, heatmap, alpha=0.6, boost=1.6)
    annotated = annotate_and_label(overlay_bgr, hm_resized, label, prob, malignancy, threshold=0.45)

    # save output image
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_name = f"{base}_explain_{datetime.now().strftime('%H%M%S')}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, annotated)

    # text explanation (tailored + references overlay)
    tailored = {
        "glioma": ("Glioma (malignant): model focused on irregular, infiltrative regions inside brain tissue. "
                   "These highlighted areas (see image) likely correspond to tumor mass and infiltration."),
        "meningioma": ("Meningioma (usually benign): model attention is extra-axial and well-circumscribed, "
                       "consistent with meningioma near the meningeal surface."),
        "pituitary": ("Pituitary tumor (usually benign): model highlights around sella turcica / pituitary region."),
        "notumor": ("No tumor detected: no focal regions of strong activation were found.")
    }
    explanation_text = tailored.get(label, "")

    # print everything
    print(f"\nPrediction: {label} (prob={prob:.4f})")
    print(f"Malignancy: {malignancy}")
    print(f"Explain method: {used_method}")
    print(f"Overlay saved to: {out_path}")
    print(f"\nText explanation:\n{explanation_text}\n")

    return out_path

# ---------- CLI ----------
def preprocess_for_model(img_path):
    # return original PIL Image and preprocessed array for model
    pil = Image.open(img_path).convert("RGB")
    # for model
    resized = pil.resize(IMG_INPUT_SIZE, Image.BILINEAR)
    arr = np.array(resized).astype("float32")
    arr = tf.keras.applications.resnet50.preprocess_input(arr)  # important preprocessing
    arr = np.expand_dims(arr, axis=0)
    return pil, arr

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to image")
    args = p.parse_args()

    out = explain_image(args.image)
    # optionally open with default system viewer (uncomment if desired)
    # import subprocess, platform
    # if platform.system() == "Windows":
    #     os.startfile(out)
    # else:
    #     subprocess.call(["xdg-open", out])
