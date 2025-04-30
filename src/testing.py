import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Setup output directories
model_dir = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_dir, exist_ok=True)

def save_report(report, filename):
    with open(os.path.join(model_dir, filename), 'w') as f:
        f.write(report)

# 2. Load and preprocess data
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("testing.csv")

X_train_raw = train_data.iloc[:, :-1].values
y_train_raw = train_data.iloc[:, -1].values
X_test_raw = test_data.iloc[:, :-1].values
y_test_raw = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train_raw = scaler.fit_transform(X_train_raw)
X_test_raw = scaler.transform(X_test_raw)

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

# 3. Model definition
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(class_id=1, name='recall'),
            tf.keras.metrics.Precision(class_id=1, name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

model = create_model(input_shape=X_train.shape[1:])

# Calculate class weights correctly
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Training callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'best_model.h5'),
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
]

# Split validation set
from sklearn.model_selection import train_test_split
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_cat, test_size=0.2, random_state=42, stratify=y_train_cat
)

history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# 4. Evaluate original model on full test set
print("\nEvaluating Original Model...")
results = model.evaluate(X_test, y_test_cat, verbose=0)
y_pred_original = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test_cat, axis=1)

original_report = classification_report(y_true, y_pred_original, target_names=["No Fall", "Fall"])
print("Original Model Classification Report:")
print(original_report)
save_report(original_report, "original_model_report.txt")

# 5. Convert to TFLite
def representative_dataset_gen():
    for i in range(min(300, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_disable_per_channel = True

tflite_model = converter.convert()

# Save TFLite model
tflite_path = os.path.join(model_dir, "model.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"\nTFLite model saved to {tflite_path}")

# 6. Evaluate TFLite model on full test set
print("\nEvaluating TFLite Model...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

y_pred_tflite = []
y_scores_tflite = []
for x in X_test:
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x, axis=0).astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred_tflite.append(np.argmax(output))
    y_scores_tflite.append(output[0][1])  # Fall probability

tflite_report = classification_report(y_true, y_pred_tflite, target_names=["No Fall", "Fall"])
print("TFLite Model Classification Report (Default Threshold 0.5):")
print(tflite_report)
save_report(tflite_report, "tflite_model_report.txt")

# 7. Calculate and save optimal threshold
precision, recall, thresholds = precision_recall_curve(y_true, y_scores_tflite)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

threshold_info = f"""Optimal Threshold: {optimal_threshold:.4f}
(Default threshold is 0.5)

To use in Wear OS:
val optimalThreshold = {optimal_threshold:.4}f
val isFall = output[0][1] > optimalThreshold
"""

with open(os.path.join(model_dir, "threshold_info.txt"), 'w') as f:
    f.write(threshold_info)

print("\nThreshold Information:")
print(threshold_info)

# 8. Save preprocessing parameters
np.save(os.path.join(model_dir, "scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(model_dir, "scaler_scale.npy"), scaler.scale_)

# 9. Save confusion matrices
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
plt.title("TFLite Model (Threshold=0.5)")

plt.suptitle("Confusion Matrices")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrices.png"))
plt.close()

print(f"\nAll outputs saved in: {os.path.abspath(model_dir)}")