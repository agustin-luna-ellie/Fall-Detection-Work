import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 1. Setup output directories
model_dir = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_dir, exist_ok=True)

def save_report(report, filename):
    with open(os.path.join(model_dir, filename), 'w') as f:
        f.write(report)

# 2. Load data (no scaling)
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("testing.csv")

X_train_raw = train_data.iloc[:, :-1].values
y_train_raw = train_data.iloc[:, -1].values
X_test_raw = test_data.iloc[:, :-1].values
y_test_raw = test_data.iloc[:, -1].values

SEQ_LENGTH = 40

def create_sequences(data, labels, seq_length):
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
        seq_labels.append(labels[i + seq_length - 1])
    return np.array(sequences), np.array(seq_labels)

X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LENGTH)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LENGTH)

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 3. Model with recall focus
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LENGTH, 3), name='input_layer'),
        tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax', name='output_layer')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(class_id=1, name='recall'),
            tf.keras.metrics.Precision(class_id=1, name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

model = create_model()

# 4. Aggressive class weights for recall
fall_weight = len(y_train) / (2 * np.sum(y_train))
class_weights = {0: 1.0, 1: fall_weight * 3.0}  # Triple weight for falls

# 5. Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=10, mode='max', restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# 6. Training
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluate original model
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_original = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

original_report = classification_report(y_true, y_pred_original, target_names=["No Fall", "Fall"])
print("Original Model Report:")
print(original_report)
save_report(original_report, "original_report.txt")

# 8. Convert to TFLite with proper quantization
def representative_dataset_gen():
    for i in range(min(300, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Quantize input
converter.inference_output_type = tf.int8  # Quantize output

# Set input/output ranges for full quantization
def set_quantization_params():
    input_details = converter._keras_model.inputs[0]
    input_min, input_max = np.min(X_train), np.max(X_train)
    converter.quantized_input_stats = {
        input_details.name: (input_min, input_max)
    }
set_quantization_params()

tflite_model = converter.convert()
tflite_path = os.path.join(model_dir, "model.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# 9. Evaluate TFLite model with proper input type
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Quantization parameters
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

y_pred_tflite = []
y_scores_tflite = []
for x in X_test:
    # Quantize input
    x_quantized = x / input_scale + input_zero_point
    x_quantized = x_quantized.astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_quantized, axis=0))
    interpreter.invoke()
    
    # Dequantize output
    output = interpreter.get_tensor(output_details[0]['index'])
    output_dequantized = (output.astype(np.float32) - output_zero_point) * output_scale
    
    y_pred_tflite.append(np.argmax(output_dequantized))
    y_scores_tflite.append(output_dequantized[0][1])

tflite_report = classification_report(y_true, y_pred_tflite, target_names=["No Fall", "Fall"])
print("\nTFLite Model Report:")
print(tflite_report)
save_report(tflite_report, "tflite_report.txt")

# 10. Precision-Recall Tradeoff
precision, recall, thresholds = precision_recall_curve(y_true, y_scores_tflite)
optimal_idx = np.argmax(recall * (precision > 0.8))  # Prioritize recall while precision > 80%
optimal_threshold = thresholds[optimal_idx]

threshold_info = f"""Optimal Threshold: {optimal_threshold:.4f}
(Default threshold is 0.5)

To use in Wear OS:
val optimalThreshold = {optimal_threshold:.4}f
val isFall = output[0][1] > optimalThreshold

Quantization Parameters:
- Input scale: {input_scale}
- Input zero point: {input_zero_point}
- Output scale: {output_scale}
- Output zero point: {output_zero_point}
"""
with open(os.path.join(model_dir, "threshold_info.txt"), 'w') as f:
    f.write(threshold_info)

# 11. Save confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm_original = confusion_matrix(y_true, y_pred_original)
sns.heatmap(cm_original, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title("Original Model")

plt.subplot(1, 2, 2)
cm_tflite = confusion_matrix(y_true, y_pred_tflite)
sns.heatmap(cm_tflite, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title("TFLite Model")

plt.suptitle("Confusion Matrices")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrices.png"))
plt.close()

print(f"\nAll outputs saved in: {os.path.abspath(model_dir)}")
print(f"TFLite model size: {len(tflite_model)/1024:.2f} KB")