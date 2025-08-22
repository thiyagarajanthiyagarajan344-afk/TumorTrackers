import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load your model
model_file = "model.keras"
model = load_model(model_file)

# Tumor -> malignant/benign mapping
malignancy_map = {
    "glioma": "malignant",
    "meningioma": "benign",
    "pituitary": "benign",
    "notumour": "benign"
}

# Prediction + explanation function
def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_pre = np.expand_dims(img_array, axis=0)
    img_pre = tf.keras.applications.resnet50.preprocess_input(img_pre)

    # Predict
    preds = model.predict(img_pre)
    class_idx = np.argmax(preds[0])
    class_labels = ["glioma", "meningioma", "notumour", "pituitary"]
    tumor_label = class_labels[class_idx]

    # Map to benign/malignant
    malignancy = malignancy_map[tumor_label]

    # Input-gradient explanation (fallback if Grad-CAM fails)
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_pre, tf.float32)
        tape.watch(inputs)
        predictions = model(inputs)
        loss = predictions[0, class_idx]
    grads = tape.gradient(loss, inputs)[0].numpy()

    # Normalize and save saliency map
    grads -= grads.min()
    grads /= grads.max() + 1e-8
    plt.imshow(grads)
    out_dir = "explanations_debug"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.basename(img_path).replace(".jpg", "_input_grad_saliency.png"))
    plt.axis('off')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

    return {
        "tumor_label": tumor_label,
        "malignancy": malignancy,
        "probability": float(preds[0, class_idx]),
        "explanation_file": out_file
    }

# CLI-like usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    result = predict_image(args.image)
    print("Prediction:", result["tumor_label"], f"(prob={result['probability']:.4f})")
    print("Malignancy:", result["malignancy"])
    print("Explanation saved to:", result["explanation_file"])
