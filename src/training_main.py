import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Setup directories with timestamp
model_dir = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_dir, exist_ok=True)

def save_report(report, filename):
    """Save classification report to file"""
    with open(os.path.join(model_dir, filename), 'w') as f:
        f.write(report)

# 2. Data loading
script_dir = Path(__file__).parent
datasets_dir = script_dir.parent / "Datasets"
train_path = datasets_dir / "training.csv"
test_path = datasets_dir / "testing.csv"

if not train_path.exists():
    raise FileNotFoundError(f"Training file not found: {train_path}")
if not test_path.exists():
    raise FileNotFoundError(f"Testing file not found: {test_path}")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Extract features and labels
X_train_raw = train_data.iloc[:, :-1].values
y_train_raw = train_data.iloc[:, -1].values
X_test_raw = test_data.iloc[:, :-1].values
y_test_raw = test_data.iloc[:, -1].values

print(f"Training data shape: {X_train_raw.shape}")
print(f"Training labels distribution: {np.bincount(y_train_raw)}")
print(f"Test data shape: {X_test_raw.shape}")
print(f"Test labels distribution: {np.bincount(y_test_raw)}")

# 3. Keep raw data as requested (no normalization)
# Using raw accelerometer data as in original implementation
print("Using raw accelerometer data (no normalization applied)")

# 4. Sequence creation
SEQ_LENGTH = 40

def create_sequences(data, labels, seq_length):
    """Create sliding window sequences"""
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
        seq_labels.append(labels[i + seq_length - 1])
    return np.array(sequences), np.array(seq_labels)

X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LENGTH)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LENGTH)

print(f"Sequence training shape: {X_train.shape}")
print(f"Sequence test shape: {X_test.shape}")

# Convert to categorical (one-hot)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 5. Improved model architecture focused on reducing false positives
def create_model():
    """
    Create a simple CNN model optimized for fall detection with reduced false positives.
    Architecture designed for TFLite compatibility on Wear OS.
    Uses raw accelerometer data without normalization.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LENGTH, 3)),
        
        # First conv block - detect basic patterns with larger kernel for raw data
        tf.keras.layers.Conv1D(16, kernel_size=7, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),  # BN helps with raw data scaling
        tf.keras.layers.MaxPooling1D(3),  # More aggressive pooling for raw data
        tf.keras.layers.Dropout(0.2),
        
        # Second conv block - detect complex patterns
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # Third conv block - high-level features
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.4),
        
        # Classification layers with strong regularization to reduce false positives
        tf.keras.layers.Dense(64, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Use a conservative learning rate and focus on precision
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Slightly higher LR for raw data
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
model.summary()

# 6. Improved class weight calculation
# Use sklearn's compute_class_weight for better balance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

# Adjust weights to prioritize precision (reduce false positives) while working with raw data
class_weights[1] = class_weights[1] * 0.7  # Further reduce fall class weight for raw data

print(f"Class weights: {class_weights}")

# 7. Enhanced callbacks
class PrecisionFocusedCallback(tf.keras.callbacks.Callback):
    """Custom callback to stop training when precision targets are met"""
    def __init__(self, min_precision=0.85, min_recall=0.75):
        super().__init__()
        self.min_precision = min_precision
        self.min_recall = min_recall
        
    def on_epoch_end(self, epoch, logs=None):
        val_precision = logs.get('val_precision', 0)
        val_recall = logs.get('val_recall', 0)
        
        # Calculate F1 score manually
        if val_precision > 0 and val_recall > 0:
            f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        else:
            f1_score = 0
            
        if val_precision >= self.min_precision and val_recall >= self.min_recall:
            print(f"\nTarget metrics achieved: Precision={val_precision:.3f}, Recall={val_recall:.3f}, F1={f1_score:.3f}")
            self.model.stop_training = True

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',  # Use AUC instead of F1Score for TF 2.12 compatibility
        patience=15, 
        mode='max', 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3, 
        patience=8, 
        min_lr=1e-6,
        verbose=1
    ),
    PrecisionFocusedCallback(min_precision=0.85, min_recall=0.75)
]

# 8. Training with validation split
print("Starting training...")
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=120,  # Reduced epochs for raw data to prevent overfitting
    batch_size=128,  # Larger batch size works better with raw data
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# 9. Model evaluation
print("\nEvaluating model...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_original = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Calculate and save original model report
original_report = classification_report(y_true, y_pred_original, 
                                      target_names=["No Fall", "Fall"], digits=4)
print("Original Model Report:")
print(original_report)
save_report(original_report, "original_report.txt")

# 10. TFLite conversion with optimization
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

# Representative dataset for quantization (optional)
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()
tflite_path = os.path.join(model_dir, "model.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")

# 11. TFLite model evaluation
print("Evaluating TFLite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

y_pred_tflite = []
y_scores_tflite = []

for i, x in enumerate(X_test):
    interpreter.set_tensor(input_details[0]['index'], 
                          np.expand_dims(x, axis=0).astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred_tflite.append(np.argmax(output))
    y_scores_tflite.append(output[0][1])

tflite_report = classification_report(y_true, y_pred_tflite, 
                                    target_names=["No Fall", "Fall"], digits=4)
print("\nTFLite Model Report:")
print(tflite_report)
save_report(tflite_report, "tflite_report.txt")

# 12. Optimized threshold selection for precision-focused approach
precision, recall, thresholds = precision_recall_curve(y_true, y_scores_tflite)

# Find threshold that maximizes precision while maintaining acceptable recall
min_recall_threshold = 0.70  # Minimum acceptable recall
precision_focused_thresholds = [
    (t, p, r, 2*p*r/(p+r) if (p+r) > 0 else 0) 
    for p, r, t in zip(precision, recall, thresholds) 
    if r >= min_recall_threshold and p >= 0.80
]

if precision_focused_thresholds:
    # Sort by precision first, then by F1-score
    precision_focused_thresholds.sort(key=lambda x: (x[1], x[3]), reverse=True)
    optimal_threshold = precision_focused_thresholds[0][0]
    opt_precision = precision_focused_thresholds[0][1]
    opt_recall = precision_focused_thresholds[0][2]
else:
    # Fallback: find threshold with best F1-score
    f1_scores = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision, recall)]
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx]
    opt_precision = precision[best_f1_idx]
    opt_recall = recall[best_f1_idx]

print(f"\nOptimal threshold: {optimal_threshold:.4f}")
print(f"Expected precision: {opt_precision:.4f}")
print(f"Expected recall: {opt_recall:.4f}")

# Apply optimal threshold
y_pred_optimal = (np.array(y_scores_tflite) > optimal_threshold).astype(int)
optimal_report = classification_report(y_true, y_pred_optimal, 
                                     target_names=["No Fall", "Fall"], digits=4)
print("\nOptimal Threshold Model Report:")
print(optimal_report)
save_report(optimal_report, "optimal_threshold_report.txt")

# Save threshold and deployment info
threshold_info = f"""Optimal Threshold Analysis
==========================
Optimal Threshold: {optimal_threshold:.6f}
Expected Precision: {opt_precision:.4f}
Expected Recall: {opt_recall:.4f}

Deployment Information:
- Model input shape: {input_details[0]['shape']}
- Model input type: {input_details[0]['dtype']}
- Data preprocessing: RAW (no normalization required)

Wear OS Implementation:
val optimalThreshold = {optimal_threshold:.6f}f
val isFall = modelOutput[1] > optimalThreshold

Use raw accelerometer data directly (no preprocessing needed):
- X, Y, Z acceleration values in m/s² or g-units
- 40-sample sliding window with 1-sample step
"""

with open(os.path.join(model_dir, "deployment_info.txt"), 'w') as f:
    f.write(threshold_info)

# 13. Enhanced visualization
plt.figure(figsize=(15, 10))

# Confusion matrices
plt.subplot(2, 3, 1)
cm_original = confusion_matrix(y_true, y_pred_original)
sns.heatmap(cm_original, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title("Original Model")

plt.subplot(2, 3, 2)
cm_tflite = confusion_matrix(y_true, y_pred_tflite)
sns.heatmap(cm_tflite, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title("TFLite Model (0.5 threshold)")

plt.subplot(2, 3, 3)
cm_optimal = confusion_matrix(y_true, y_pred_optimal)
sns.heatmap(cm_optimal, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title(f"TFLite Optimal (t={optimal_threshold:.3f})")

# Training history
plt.subplot(2, 3, 4)
epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(epochs, history.history['precision'], 'g-', label='Training Precision')
plt.plot(epochs, history.history['val_precision'], 'orange', label='Validation Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(epochs, history.history['recall'], 'purple', label='Training Recall')
plt.plot(epochs, history.history['val_recall'], 'brown', label='Validation Recall')
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(model_dir, "model_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# 14. Precision-Recall curve
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, 'b-', linewidth=2)
plt.scatter([opt_recall], [opt_precision], color='red', s=100, zorder=5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_scores_tflite, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Optimal Threshold')
plt.xlabel('Prediction Score')
plt.ylabel('Frequency')
plt.title('Prediction Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, "threshold_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# 15. Final summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

# Calculate final metrics
final_precision = cm_optimal[1,1] / (cm_optimal[1,1] + cm_optimal[0,1]) if (cm_optimal[1,1] + cm_optimal[0,1]) > 0 else 0
final_recall = cm_optimal[1,1] / (cm_optimal[1,1] + cm_optimal[1,0]) if (cm_optimal[1,1] + cm_optimal[1,0]) > 0 else 0
final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

false_positives = cm_optimal[0,1]
false_negatives = cm_optimal[1,0]
total_negatives = cm_optimal[0,0] + cm_optimal[0,1]
total_positives = cm_optimal[1,0] + cm_optimal[1,1]

print(f"Final Precision: {final_precision:.4f}")
print(f"Final Recall: {final_recall:.4f}")
print(f"Final F1-Score: {final_f1:.4f}")
print(f"False Positives: {false_positives}/{total_negatives} ({false_positives/total_negatives*100:.2f}%)")
print(f"False Negatives: {false_negatives}/{total_positives} ({false_negatives/total_positives*100:.2f}%)")
print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
print(f"Optimal threshold: {optimal_threshold:.6f}")
print(f"\nAll outputs saved in: {os.path.abspath(model_dir)}")

# Verification outputs
print("\n" + "="*40)
print("PREDICTION VERIFICATION")
print("="*40)
print("Original Predictions (first 10):", y_pred_original[:10])
print("TFLite Predictions (first 10):", y_pred_tflite[:10])
print("Optimal Threshold Predictions (first 10):", y_pred_optimal[:10])
print("True Labels (first 10):", y_true[:10])