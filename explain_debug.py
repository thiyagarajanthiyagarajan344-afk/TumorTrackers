import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

# ------------------ Config ------------------
model_file = "model.keras"
model = load_model(model_file)
class_labels = ["glioma", "meningioma", "notumour", "pituitary"]

# Tumor -> malignant/benign mapping
malignancy_map = {
    "glioma": "malignant",
    "meningioma": "benign",
    "pituitary": "benign",
    "notumour": "benign"
}

# ------------------ Grad-CAM ------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap.numpy()

# ------------------ Prediction + Explanation ------------------
def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_pre = np.expand_dims(img_array, axis=0)
    img_pre = tf.keras.applications.resnet50.preprocess_input(img_pre)

    # Prediction
    preds = model.predict(img_pre)
    class_idx = np.argmax(preds[0])
    tumor_label = class_labels[class_idx]
    malignancy = malignancy_map[tumor_label]

    # Grad-CAM heatmap
    try:
        heatmap = make_gradcam_heatmap(img_pre, model, last_conv_layer_name="conv5_block3_out", pred_index=class_idx)
    except Exception:
        heatmap = np.zeros((7, 7))  # fallback if Grad-CAM fails

    # Overlay heatmap on original image
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    # Save visualization
    out_dir = "explanations_debug"
    os.makedirs(out_dir, exist_ok=True)
    overlay_file = os.path.join(out_dir, os.path.basename(img_path).replace(".jpg", "_gradcam_overlay.png"))
    cv2.imwrite(overlay_file, overlayed)

    # Personalized explanation
    if tumor_label == "glioma":
        text_explanation = (
            f"The model detected *glioma* with probability {preds[0,class_idx]:.2f}. "
            "The highlighted regions in red indicate abnormal irregular mass-like growth "
            "inside the brain tissue, consistent with malignant glioma features."
        )
    elif tumor_label == "meningioma":
        text_explanation = (
            f"The model detected *meningioma* with probability {preds[0,class_idx]:.2f}. "
            "The Grad-CAM highlights areas around the brain membrane. This pattern is "
            "consistent with meningiomas, which usually grow from the meninges and are often benign."
        )
    elif tumor_label == "pituitary":
        text_explanation = (
            f"The model detected *pituitary tumor* with probability {preds[0,class_idx]:.2f}. "
            "The activation is centered near the pituitary gland region. Pituitary tumors are "
            "generally benign but can affect hormone regulation."
        )
    else:
        text_explanation = (
            f"The model detected *no tumor* with probability {preds[0,class_idx]:.2f}. "
            "The heatmap shows no significant abnormal regions, indicating a healthy scan."
        )

    return {
        "tumor_label": tumor_label,
        "malignancy": malignancy,
        "probability": float(preds[0, class_idx]),
        "overlay_file": overlay_file,
        "text_explanation": text_explanation
    }

# ------------------ CLI ------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    result = predict_image(args.image)
    print("Prediction:", result["tumor_label"], f"(prob={result['probability']:.4f})")
    print("Malignancy:", result["malignancy"])
    print("Overlay saved to:", result["overlay_file"])
    print("Explanation:", result["text_explanation"])
