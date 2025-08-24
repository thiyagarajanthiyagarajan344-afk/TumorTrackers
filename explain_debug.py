import os
import json
import math
import argparse
from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# --------------------------- Config Defaults ---------------------------
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # must match your explainer
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_FREEZE = 12           # stage 1: train top
EPOCHS_FT = 12               # stage 2: fine-tune backbone
INIT_LR = 1e-4
FT_LR = 2e-5
VAL_SPLIT = 0.15
MODEL_FILE = "model.keras"
LABELS_FILE = "class_names.json"
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# --------------------------- Utils ---------------------------
def set_mixed_precision(enable=True):
    if enable:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled.")
        except Exception as e:
            print(f"[WARN] Mixed precision not enabled: {e}")

def fix_seeds(seed=SEED):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def list_counts_per_class(train_dir, class_names):
    counts = {}
    for cname in class_names:
        cdir = os.path.join(train_dir, cname)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        total = sum(1 for root, _, files in os.walk(cdir) for f in files if f.lower().endswith((".png",".jpg",".jpeg",".bmp")))
        counts[cname] = total
    return counts

def compute_class_weights(counts_dict):
    # Inverse frequency class weights: weight_c = N_total / (num_classes * count_c)
    total = sum(counts_dict.values())
    num_classes = len(counts_dict)
    weights = {}
    for i, cname in enumerate(CLASS_NAMES):
        c = counts_dict[cname]
        weights[i] = (total / (num_classes * max(c, 1)))
    return weights

def make_datasets(data_dir, img_size, batch_size, val_split, class_names, seed):
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")  # optional

    # Force class order to match CLASS_NAMES for consistency with explainer
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
        class_names=class_names,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="validation",
        class_names=class_names,
    )

    test_ds = None
    if os.path.isdir(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=(img_size, img_size),
            shuffle=False,
            class_names=class_names,
        )

    # Cache + prefetch for performance
    def prep(ds, training=False):
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        if training:
            ds = ds.shuffle(1024, seed=seed)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    return prep(train_ds, training=True), prep(val_ds, training=False), (prep(test_ds, training=False) if test_ds else None)

def build_model(img_size, num_classes, dropout=0.4, train_base=False):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Data augmentation inside the graph
    aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )

    x = aug(inputs)
    x = layers.Lambda(effnet_preprocess, name="preprocess")(x)

    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = train_base  # stage 1: False; stage 2: True (partial)

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    # Important: for mixed_precision, final dtype float32
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="BrainTumor_EfficientNetB0")
    return model, base

def compile_model(model, lr):
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc", multi_label=False, num_thresholds=200),
        ],
    )

def fine_tune_setup(base_model, num_unfreeze=80):
    # Unfreeze top N layers of EfficientNet for fine-tuning
    total = len(base_model.layers)
    unfreeze_from = max(0, total - num_unfreeze)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= unfreeze_from)

    print(f"[INFO] Fine-tuning enabled. Unfrozen layers from index {unfreeze_from}/{total} (~{num_unfreeze} layers).")

def train_and_finetune(
    data_dir,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    init_lr=INIT_LR,
    ft_lr=FT_LR,
    epochs_freeze=EPOCHS_FREEZE,
    epochs_ft=EPOCHS_FT,
    val_split=VAL_SPLIT,
    model_file=MODEL_FILE,
    enable_mixed_precision=True,
):
    # Determinism
    fix_seeds(SEED)
    set_mixed_precision(enable_mixed_precision)

    # Datasets
    train_ds, val_ds, test_ds = make_datasets(data_dir, img_size, batch_size, val_split, CLASS_NAMES, SEED)

    # Class weights from directory counts (Training)
    counts = list_counts_per_class(os.path.join(data_dir, "Training"), CLASS_NAMES)
    print("[INFO] Training image counts per class:", counts)
    class_weights = compute_class_weights(counts)
    print("[INFO] Computed class weights:", class_weights)

    # Build model (stage 1)
    model, base = build_model(img_size, num_classes=len(CLASS_NAMES), dropout=0.4, train_base=False)
    compile_model(model, init_lr)

    # Callbacks
    ckpt = ModelCheckpoint(model_file, monitor="val_loss", save_best_only=True, verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    # Stage 1: Train head
    print("\n[STAGE 1] Training classification head (backbone frozen)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_freeze,
        class_weight=class_weights,
        callbacks=[ckpt, es, rlrop],
        verbose=2,
    )

    # Stage 2: Fine-tune
    print("\n[STAGE 2] Fine-tuning backbone...")
    fine_tune_setup(base, num_unfreeze=80)
    compile_model(model, ft_lr)
    es2 = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    rlrop2 = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ckpt2 = ModelCheckpoint(model_file, monitor="val_loss", save_best_only=True, verbose=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_ft,
        class_weight=class_weights,
        callbacks=[ckpt2, es2, rlrop2],
        verbose=2,
    )

    # Ensure final best model is on disk and in memory
    model.save(model_file)
    print(f"[INFO] Saved best model to {model_file}")

    # Save class name order for reference/debug
    with open(LABELS_FILE, "w") as f:
        json.dump(CLASS_NAMES, f)
    print(f"[INFO] Saved class names to {LABELS_FILE}")

    # Evaluate
    print("\n[EVAL] Validation set:")
    val_metrics = model.evaluate(val_ds, verbose=0)
    for name, val in zip(model.metrics_names, val_metrics):
        print(f"  {name}: {val:.4f}")

    if test_ds is not None:
        print("\n[EVAL] Testing set:")
        test_metrics = model.evaluate(test_ds, verbose=0)
        for name, val in zip(model.metrics_names, test_metrics):
            print(f"  {name}: {val:.4f}")

        # Optional: confusion matrix
        try:
            y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
            y_pred = []
            for x, _ in test_ds:
                prob = model.predict(x, verbose=0)
                y_pred.append(np.argmax(prob, axis=1))
            y_pred = np.concatenate(y_pred, axis=0)
            cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES)).numpy()
            print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
        except Exception as e:
            print(f"[WARN] Could not compute confusion matrix: {e}")

    print("\n[DONE] Training complete. Use model.keras with your explain script.")

# --------------------------- CLI ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train brain tumor classifier with EfficientNetB0")
    ap.add_argument("--data_dir", required=True, help="Path to dataset folder containing Training/ (and optional Testing/)")
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs_freeze", type=int, default=EPOCHS_FREEZE)
    ap.add_argument("--epochs_ft", type=int, default=EPOCHS_FT)
    ap.add_argument("--init_lr", type=float, default=INIT_LR)
    ap.add_argument("--ft_lr", type=float, default=FT_LR)
    ap.add_argument("--val_split", type=float, default=VAL_SPLIT)
    ap.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_finetune(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        init_lr=args.init_lr,
        ft_lr=args.ft_lr,
        epochs_freeze=args.epochs_freeze,
        epochs_ft=args.epochs_ft,
        val_split=args.val_split,
        enable_mixed_precision=(not args.no_mixed_precision),
    )
