# main.py (Grad-CAM fixed: conv layer output re-rooted into top-level graph)
import os
import argparse
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------- Config ---------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
MODEL_FILE = 'model.keras'
PIPELINE_FILE = 'pipeline.joblib'
AUTOTUNE = tf.data.AUTOTUNE

# ------------------------- Helpers ----------------------------
def get_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_dir = os.path.join(data_dir, 'Training')
    test_dir  = os.path.join(data_dir, 'Testing')

    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(f"Expected Training/ and Testing/ inside {data_dir}")

    image_ds_from_dir = keras.utils.image_dataset_from_directory
    train_ds = image_ds_from_dir(train_dir, label_mode='int', image_size=(img_size, img_size),
                                 batch_size=batch_size, shuffle=True)
    val_ds = image_ds_from_dir(test_dir, label_mode='int', image_size=(img_size, img_size),
                               batch_size=batch_size, shuffle=False)

    data_augmentation = keras.Sequential([layers.RandomFlip('horizontal'), layers.RandomRotation(0.05)])
    preprocess_fn = tf.keras.applications.resnet50.preprocess_input

    train_ds = (train_ds
                .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
                .map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds = (val_ds
              .map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
              .prefetch(AUTOTUNE))

    classes = sorted([d.name for d in Path(train_dir).iterdir() if d.is_dir()])
    return train_ds, val_ds, classes

# --------------------------- Model ----------------------------
def build_model(num_classes, img_size=IMG_SIZE, lr=LEARNING_RATE):
    base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                         input_shape=(img_size, img_size, 3), name='resnet50')
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3), name='input_layer')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)
    model = keras.Model(inputs, outputs, name='brain_tumor_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------- Training --------------------------
def train(data_dir, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    train_ds, val_ds, class_names = get_datasets(data_dir, img_size, batch_size)
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes in Training/")
    model = build_model(num_classes, img_size)
    ckpt = keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', save_best_only=True)
    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[ckpt, es])
    pipeline = {'class_names': class_names, 'img_size': img_size, 'preprocess': 'resnet50'}
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Training finished. Saved model and pipeline.")

# ------------------------ Grad-CAM Utils (final fix) -----------------------
def find_resnet_and_conv_layer(model, prefer_name='conv5_block3_out'):
    # Prefer the usual ResNet50 final conv block
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name == 'resnet50':
            names = [l.name for l in layer.layers]
            if prefer_name in names:
                return layer, prefer_name
            for nl in reversed(layer.layers):
                if isinstance(nl, layers.Conv2D):
                    return layer, nl.name
    # fallback: top-level
    for nl in reversed(model.layers):
        if isinstance(nl, layers.Conv2D):
            return model, nl.name
    raise ValueError("No Conv2D layer found in model or nested models.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Build a grad_model where the conv-layer output is re-rooted into the top-level graph.
    This ensures the conv activations are the exact activations that produce model.output.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # locate resnet nested model and conv name
    if last_conv_layer_name is None:
        resnet_model, conv_name = find_resnet_and_conv_layer(model)
    else:
        # try to find conv in nested resnet first
        resnet_model = None
        conv_name = last_conv_layer_name
        for layer in model.layers:
            if isinstance(layer, keras.Model) and layer.name == 'resnet50':
                try:
                    _ = layer.get_layer(conv_name)
                    resnet_model = layer
                    break
                except Exception:
                    pass
        if resnet_model is None:
            try:
                _ = model.get_layer(conv_name)
                resnet_model = model
            except Exception:
                resnet_model, conv_name = find_resnet_and_conv_layer(model)

    # Build conv_model that maps resnet.input -> conv layer output (symbolic inside resnet)
    conv_layer_obj = resnet_model.get_layer(conv_name)
    conv_model = tf.keras.Model(inputs=resnet_model.input, outputs=conv_layer_obj.output)

    # Now apply conv_model to the top-level model's input symbolic tensor to get a symbolic
    # conv activation that lives in the top-level graph:
    # (this "re-roots" the conv-layer output into the same graph as model.output)
    conv_on_top = conv_model(model.inputs[0])   # symbolic KerasTensor in top graph

    # Build grad_model that maps top-level inputs -> [conv_on_top, model.output]
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_on_top, model.output])

    # compute numeric outputs and gradients on numeric tensor
    # Call grad_model on numeric tensor. Some Keras versions accept tensor directly;
    # we'll try both forms to be robust.
    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            # ensure the tape will compute grads wrt conv_outputs
            tape.watch(conv_outputs)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
    except Exception:
        # fallback: try passing a list (structure matching model.inputs)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([img_tensor], training=False)
            tape.watch(conv_outputs)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Gradients are None — conv_outputs may not be connected to predictions. "
            "This should not happen with the re-rooted grad_model. "
            "If it does, please paste the output of model.summary() and the last 20 "
            "layer names of the nested resnet (I'll inspect them)."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image_pil(orig_img_pil, heatmap, alpha=0.4):
    img = np.array(orig_img_pil.convert('RGB'))
    hmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hmap_uint8 = np.uint8(255 * hmap)
    colored = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)  # BGR
    overlay = cv2.addWeighted(colored, alpha, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

# ------------------------ Prediction Flow ---------------------
def load_pipeline(pipeline_file=PIPELINE_FILE):
    pipeline = joblib.load(pipeline_file)
    preprocess_fn = tf.keras.applications.resnet50.preprocess_input
    return pipeline, preprocess_fn

def predict_and_explain(image_path, model_file=MODEL_FILE, pipeline_file=PIPELINE_FILE, output_dir='explanations'):
    os.makedirs(output_dir, exist_ok=True)
    pipeline, preprocess_fn = load_pipeline(pipeline_file)
    class_names = pipeline['class_names']
    img_size = pipeline['img_size']

    model = keras.models.load_model(model_file)
    model.trainable = False

    orig = Image.open(image_path).convert('RGB')
    img_resized = orig.resize((img_size, img_size))
    img_array = np.array(img_resized).astype('float32')
    img_pre = preprocess_fn(np.expand_dims(img_array, axis=0))  # (1,H,W,3)

    preds = model.predict(img_pre)
    pred_idx = int(np.argmax(preds[0]))
    pred_label = class_names[pred_idx]
    pred_prob = float(preds[0, pred_idx])

    # Get the conv layer (prefer conv5_block3_out) and compute Grad-CAM
    resnet_model, conv_name = find_resnet_and_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_pre, model, last_conv_layer_name=conv_name, pred_index=pred_idx)

    overlay_np = overlay_heatmap_on_image_pil(orig, heatmap)
    if overlay_np.dtype != np.uint8:
        overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)

    overlay_pil = Image.fromarray(overlay_np)
    base_name = Path(image_path).stem
    overlay_path = os.path.join(output_dir, f'{base_name}_gradcam.png')
    orig_path = os.path.join(output_dir, f'{base_name}_orig.png')
    overlay_pil.save(overlay_path)
    orig.save(orig_path)

    return {
        'image': image_path,
        'pred_label': pred_label,
        'pred_prob': pred_prob,
        'overlay_path': overlay_path,
        'orig_path': orig_path,
        'conv_layer_used': conv_name,
        'conv_parent_name': resnet_model.name if hasattr(resnet_model, 'name') else str(resnet_model)
    }

# ---------------------------- CLI -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train / Predict Brain Tumor model (TensorFlow)')
    sub = parser.add_subparsers(dest='cmd')

    train_p = sub.add_parser('train')
    train_p.add_argument('--data', required=True, help='root folder containing Training/ and Testing/')
    train_p.add_argument('--epochs', type=int, default=NUM_EPOCHS)

    pred_p = sub.add_parser('predict')
    pred_p.add_argument('--image', required=True, help='image file to predict')
    pred_p.add_argument('--model', default=MODEL_FILE, help='model file (keras)')
    pred_p.add_argument('--pipeline', default=PIPELINE_FILE, help='pipeline joblib file')
    pred_p.add_argument('--out', default='explanations', help='output folder for overlays')

    args = parser.parse_args()

    if args.cmd == 'train':
        print('Training with data:', args.data)
        train(args.data, epochs=args.epochs)
    elif args.cmd == 'predict':
        print('Predicting for image:', args.image)
        out = predict_and_explain(args.image, model_file=args.model, pipeline_file=args.pipeline, output_dir=args.out)
        print('Prediction:', out['pred_label'], f"(prob={out['pred_prob']:.4f})")
        print('Saved overlay to', out['overlay_path'])
        print('Used conv layer:', out['conv_layer_used'], 'in parent model:', out['conv_parent_name'])
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
