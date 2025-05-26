import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
#mportacion de las librerias 

# 1. Hacemos un setup de directorios en donde guardamos a los modelo entitutlado con un time-stamp
model_dir = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_dir, exist_ok=True)

#Funcion para guardar los reporter (que utiliza "classification_report" de sklearn)
def save_report(report, filename):
    with open(os.path.join(model_dir, filename), 'w') as f:
        f.write(report)

# 2. Carga de datos (utilizados ambos del dataset del paper de SmartFall)

script_dir = Path(__file__).parent

# Ajustar las rutas para que apunten al nivel superior donde está la carpeta Datasets
datasets_dir = script_dir.parent / "Datasets"

# Construir rutas a los archivos
train_path = datasets_dir / "training.csv"
test_path = datasets_dir / "testing.csv"

# Verificar que existen
if not train_path.exists():
    raise FileNotFoundError(f"No se encontró {train_path}")
if not test_path.exists():
    raise FileNotFoundError(f"No se encontró {test_path}")

# Cargar los datos
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
# Obtenemos todos los datos de entrenamiento y testeo, sin escalarlos
X_train_raw = train_data.iloc[:, :-1].values
y_train_raw = train_data.iloc[:, -1].values
X_test_raw = test_data.iloc[:, :-1].values
y_test_raw = test_data.iloc[:, -1].values


# Creacion de las ventanas , donde establecemos un largo de las mismas (el sliding es de todos los datos
#excepto el primero ; si la primer ventana va de 0-5 la segunda ira de 1-6 y asi sucesivamente)
SEQ_LENGTH = 40

def create_sequences(data, labels, seq_length):
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
        seq_labels.append(labels[i + seq_length - 1])
    return np.array(sequences), np.array(seq_labels)

#Creamos las secuencias a partir de los datos
X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LENGTH)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LENGTH)

# Convertimos las etiquetas a formato one-hot encoding (0 y 1)
# (0: no fall, 1: fall)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 3. Creacion del modelo
"""Para esta parte es importnate entender las limitaciones de tflite dentro de los dispositivos de 
samsung watch wear os, ya que no soporta todas las operaciones, y en caso de requerirlas se debe utilizar
el optimizer de tflite pero este aun no esta habilitado en las bibliotecas de kotlin. Por lo tanto
no se pudo respetar la arquitectura del paper, y se opto por una arquitectura mas simple, pero que 
cumple con los objetivos. De este modo se recayo en el uso de convolucionales simple 
"""
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LENGTH, 3)),
        
        # Extraciion de caracteristicas 
        tf.keras.layers.Conv1D(16, 7, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(4),  # Pooling mas agresivo
        
        tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        
        # Analisis de patron temporal
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Clasificacion final regularizada
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    #Compilacion del modelo
    """" Se utiliza la loss de categorical-crossentropy ya que se pasaron a one-hot las etiquetas, ademas
    de esto buscamos evaluar principalmente las metricas de recall ( que son sumamente importante
    para la deteccion de las caidas) y la precision a modo de no tener una alta tasa de falsos positivos.
    Tambien se utiliza el AUC para evaluar el rendimiento del modelo y que no genere overfitting"""

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

model = create_model()

# 4. Balance de los pesos de las clases ya que hay un gran desbalance (85000 no fall y 5000 fall)
#Quizas habria que considerar usar la funcion de compute_class_weight de sklearn (tomar atencion mas adelante)
fall_weight = len(y_train) / (2 * np.sum(y_train))
class_weights = {0: 1.0, 1: fall_weight * 1.5}  

# 5. Callbacks para el control de las metricas (ATENCION SE PUEDE SACAR SIN PROBLEMA O AJUSTAR )
class BalanceMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Target: precision > 0.5 and recall between 0.7-0.9
        if logs['val_precision'] > 0.9 and 0.9 <= logs['val_recall'] <= 0.95:
            self.model.stop_training = True


#Callbacks usados en el entrenamiento
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=30, mode='max', restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    BalanceMetrics()
]

# 6. Entrenamieno
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=100,  
    batch_size=128,  
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluacion del modelo
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_original = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

#Se hace el reporte del modelo
original_report = classification_report(y_true, y_pred_original, target_names=["No Fall", "Fall"])
print("Original Model Report:")
print(original_report)
save_report(original_report, "original_report.txt")

# 8. Convertir a TFLITE (float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(model_dir, "model.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# 9. Evaluacion del TFLITE
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
    y_scores_tflite.append(output[0][1])

tflite_report = classification_report(y_true, y_pred_tflite, target_names=["No Fall", "Fall"])
print("\nTFLite Model Report:")
print(tflite_report)
save_report(tflite_report, "tflite_report.txt")

# 10. Tuning del umbral
"""Buscamos encontrar el umbral optimo con el cual el recall permanezca por encima de cierto valor
mientras maximizamos la precision del modelo"""
precision, recall, thresholds = precision_recall_curve(y_true, y_scores_tflite)

# Buscamos el umbral en donde el recall supera un cierto valor y se maximiza la precision
viable_thresholds = [(t, p, r) for p, r, t in zip(precision, recall, thresholds) if  p >= 0.95]

if viable_thresholds:
    # Ordenar por una métrica combinada (por ejemplo, F1-score)
    viable_thresholds.sort(key=lambda x: 2 * (x[1] * x[2]) / (x[1] + x[2]), reverse=True)  # F1-score
    optimal_threshold = viable_thresholds[0][0]
else:
    optimal_threshold = 0.5  # Fallback

threshold_info = f"""Optimal Threshold: {optimal_threshold:.4f}
(Default threshold is 0.5)

To use in Wear OS:
val optimalThreshold = {optimal_threshold:.4}f
val isFall = output[0][1] > optimalThreshold
"""
with open(os.path.join(model_dir, "threshold_info.txt"), 'w') as f:
    f.write(threshold_info)

# 11. Creacion de las matrices de confusion
"""Se muestran 3 matrices de confusion, la del modelo oriinal
la del modelo de tflite y la dle tflite con el umbral optimo para el funcionamiento segun las metricas
que se evaluaron previamente"""
y_pred_optimal = (np.array(y_scores_tflite) > optimal_threshold).astype(int)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
cm_original = confusion_matrix(y_true, y_pred_original)
sns.heatmap(cm_original, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title("Original Model")


plt.subplot(1, 3, 2)
cm_tflite = confusion_matrix(y_true, y_pred_tflite)
sns.heatmap(cm_tflite, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title(f"TFLite Model (Threshold=0.5")

plt.subplot(1, 3, 3)
cm_tflite = confusion_matrix(y_true, y_pred_optimal)
sns.heatmap(cm_tflite, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title(f"TFLite Model (Threshold={optimal_threshold:.2f})")

plt.suptitle("Confusion Matrices")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrices.png"))
plt.close()

print(f"\nAll outputs saved in: {os.path.abspath(model_dir)}")


# Verificar si funciona bien despues
print("Original Predictions (first 10):", y_pred_original[:10])
print("TFLite Predictions (first 10):", y_pred_tflite[:10])
print("Optimal Threshold Predictions (first 10):", y_pred_optimal[:10])