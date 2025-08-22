import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import cv2

MODEL_FILE = "model.keras"
IMG_SIZE = 224

# ------------------ Grad-CAM ------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out', pred_index=None):
    """
    img_array: preprocessed input, shape (1, H, W, 3)
    model: top-level Functional model
    last_conv_layer_name: name of last conv layer in ResNet50 base
    """
    # get the ResNet50 nested model
    resnet_layer = model.get_layer("resnet50")
    conv_layer = resnet_layer.get_layer(last_conv_layer_name)

    # new model: from top input to conv output + original output
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(colored, alpha, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay)

# ------------------ Main Prediction ------------------
def predict_image(image_path, model_file=MODEL_FILE):
    model = keras.models.load_model(model_file)
    model.trainable = False

    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype('float32')
    img_pre = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))

    preds = model.predict(img_pre)
    pred_idx = int(np.argmax(preds[0]))
    class_names = sorted([d.name for d in os.scandir("BrainTumor/BrainTumor/dataset/Training") if d.is_dir()])
    pred_label = class_names[pred_idx]
    pred_prob = float(preds[0, pred_idx])

    heatmap = make_gradcam_heatmap(img_pre, model, last_conv_layer_name='conv5_block3_out', pred_index=pred_idx)
    overlay = overlay_heatmap_on_image(image_path, heatmap)

    overlay_path = os.path.join("explanations", os.path.basename(image_path).replace(".jpg", "_gradcam.png"))
    os.makedirs("explanations", exist_ok=True)
    overlay.save(overlay_path)

    return {"label": pred_label, "prob": pred_prob, "overlay": overlay_path}

# ------------------ Run ------------------
if __name__ == "__main__":
    image_path = "BrainTumor/BrainTumor/dataset/Testing/meningioma/Te-me_0012.jpg"
    result = predict_image(image_path)
    print(f"Prediction: {result['label']} (prob={result['prob']:.4f})")
    print("Saved Grad-CAM overlay to:", result['overlay'])
