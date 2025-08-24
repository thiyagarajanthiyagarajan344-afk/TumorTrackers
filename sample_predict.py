
# sample_predict.py - exposes predict_and_explain(image_path)
import os, uuid, numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import cv2

MODEL_CANDIDATES = ["model.keras", "model.h5", "app/models/model.keras", "app/models/model.h5"]
IMG_SIZE = 224
OUT_DIR = "explanations"
os.makedirs(OUT_DIR, exist_ok=True)

def _find_model_file():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No model file found. Place your Keras model at one of: " + ", ".join(MODEL_CANDIDATES))

def load_model_once():
    global _LOADED_MODEL
    if '_LOADED_MODEL' in globals() and _LOADED_MODEL is not None:
        return _LOADED_MODEL
    model_file = _find_model_file()
    model = keras.models.load_model(model_file, compile=False)
    model.trainable = False
    _LOADED_MODEL = model
    return model

def find_resnet_and_conv_layer(model):
    for layer in model.layers:
        if isinstance(layer, keras.Model) and 'resnet' in layer.name.lower():
            for l in reversed(layer.layers):
                if isinstance(l, keras.layers.Conv2D):
                    return layer, l.name
    for l in reversed(model.layers):
        if isinstance(l, keras.layers.Conv2D):
            return model, l.name
    raise RuntimeError("No Conv2D found")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    resnet_model, conv_name = find_resnet_and_conv_layer(model) if last_conv_layer_name is None else (None, last_conv_layer_name)
    try:
        conv_layer_obj = resnet_model.get_layer(conv_name)
        conv_model = tf.keras.Model(inputs=resnet_model.input, outputs=conv_layer_obj.output)
        conv_on_top = conv_model(model.inputs[0])
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_on_top, model.output])
    except Exception:
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[model.layers[-3].output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        heatmap = heatmap.numpy()
    else:
        heatmap = (heatmap / (max_val + 1e-8)).numpy()
    return heatmap

def overlay_and_save(orig_pil, heatmap, out_path):
    img = np.array(orig_pil.convert('RGB'))
    hmap = cv2.resize((heatmap*255).astype('uint8'), (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(colored, 0.5, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    Image.fromarray(overlay).save(out_path)
    return out_path

def predict_and_explain(image_path):
    model = load_model_once()
    orig = Image.open(image_path).convert('RGB')
    resized = orig.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(resized).astype('float32')
    pre = tf.keras.applications.resnet50.preprocess_input(arr)
    pre = np.expand_dims(pre, axis=0)
    preds = model.predict(pre)
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0, idx])
    classes = None
    if os.path.isdir('BrainTumor/dataset/Training'):
        try:
            classes = sorted([d.name for d in os.scandir('BrainTumor/dataset/Training') if d.is_dir()])
        except Exception:
            classes = None
    label = classes[idx] if classes and idx < len(classes) else f'class_{idx}'
    heatmap = make_gradcam_heatmap(np.expand_dims(pre[0], axis=0), model, pred_index=idx)
    out_name = f"{Path(image_path).stem}_{uuid.uuid4().hex[:8]}_gradcam.png"
    out_path = os.path.join(OUT_DIR, out_name)
    overlay_and_save(orig, heatmap, out_path)
    return {'pred_label': label, 'pred_prob': prob, 'overlay_path': out_path}
