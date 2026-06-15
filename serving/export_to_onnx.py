"""
STEP 1 — Export lstm_epoch_05.keras → lstm_autoscaler.onnx
Run this cell in your Colab notebook after training.
"""

import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np
import shutil
import os

# ─── CONFIG ───────────────────────────────────────────────────────
DRIVE_PATH   = "/content/drive/MyDrive/Bitbrains_LSTM_Predictive_Scaler"
KERAS_FILE   = f"{DRIVE_PATH}/lstm_epoch_05.keras"
ONNX_FILE    = f"{DRIVE_PATH}/lstm_autoscaler.onnx"
LOCAL_KERAS  = "/content/lstm_epoch_05.keras"
LOCAL_ONNX   = "/content/lstm_autoscaler.onnx"
WINDOW_SIZE  = 60
N_FEATURES   = 5    # cpu, ram, diskw, t_sin, t_cos

# ─── STEP 1A — Copy from Drive to local SSD (faster) ─────────────
print("Copying model from Drive to local...")
shutil.copy(KERAS_FILE, LOCAL_KERAS)
print(f"Copied: {LOCAL_KERAS}")

# ─── STEP 1B — Load Keras model ───────────────────────────────────
def accuracy_metric(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.cast(diff < 0.10, tf.float32))

print("Loading Keras model...")
model = tf.keras.models.load_model(
    LOCAL_KERAS,
    custom_objects={'accuracy_metric': accuracy_metric},
    compile=False
)
model.summary()

# ─── STEP 1C — Verify input/output shapes ────────────────────────
print(f"\nInput  shape: {model.input_shape}")   # should be (None, 60, 5)
print(f"Output shape: {model.output_shape}")    # should be (None, 3)
assert model.input_shape  == (None, WINDOW_SIZE, N_FEATURES), \
    f"Expected input (None,60,5) got {model.input_shape}"
assert model.output_shape == (None, 3), \
    f"Expected output (None,3) got {model.output_shape}"
print("Shape check passed!")

# ─── STEP 1D — Convert to ONNX ───────────────────────────────────
print("\nConverting to ONNX...")

@tf.function(input_signature=[
    tf.TensorSpec((None, WINDOW_SIZE, N_FEATURES), tf.float32, name="input")
])
def serving_fn(inputs):
    return model(inputs)

tf2onnx.convert.from_function(
    serving_fn,
    input_signature=[tf.TensorSpec((None, WINDOW_SIZE, N_FEATURES), tf.float32)],
    opset=13,
    output_path=LOCAL_ONNX
)
print(f"ONNX saved: {LOCAL_ONNX}")

# ─── STEP 1E — Verify ONNX with dummy input ──────────────────────
print("\nVerifying ONNX model...")
sess       = ort.InferenceSession(LOCAL_ONNX)
input_name = sess.get_inputs()[0].name
dummy      = np.random.rand(1, WINDOW_SIZE, N_FEATURES).astype(np.float32)
output     = sess.run(None, {input_name: dummy})[0]

print(f"ONNX input  shape: {sess.get_inputs()[0].shape}")
print(f"ONNX output shape: {output.shape}")   # should be (1, 3)
print(f"Sample prediction: CPU={output[0][0]*100:.1f}%  RAM={output[0][1]*100:.1f}%  Disk={output[0][2]*100:.1f}%")
print("ONNX verification passed!")

# ─── STEP 1F — Copy ONNX back to Drive ───────────────────────────
print("\nSaving ONNX to Google Drive...")
shutil.copy(LOCAL_ONNX, ONNX_FILE)
print(f"Saved to Drive: {ONNX_FILE}")
print("\nSTEP 1 COMPLETE — Download lstm_autoscaler.onnx from Drive to your local machine.")
