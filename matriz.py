import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


# Carregar o modelo previamente treinado
model = tf.keras.models.load_model('model/modelo3.h5')

# Definir classes
classes = ['NORMAL', 'DRUSAS', 'DME', 'NVC']

# Função para carregar as imagens de validação
def load_validation_data(directory):
    images = []
    labels = []
    for class_folder in os.listdir(directory):
        class_label = int(class_folder)  # Convertendo o nome da subpasta para inteiro
        class_folder_path = os.path.join(directory, class_folder)
        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(496, 512))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0  # Normalizar os pixels
            images.append(image)
            labels.append(class_label)
    return np.array(images), np.array(labels)

# Carregar os dados de validação
X_val, y_val = load_validation_data('val_new')

# Prever classes para as imagens de validação
y_pred = np.argmax(model.predict(X_val), axis=1)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_val, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calcular a acuracia
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print("Accuracy:", accuracy)

# Calcular o F1-score
f1score = f1_score(y_val, y_pred, average='weighted')
print("F1-score:", f1score)

# Imprimir a precisão para cada classe
precision_per_class = precision_score(y_val, y_pred, average=None)
for i in range(len(precision_per_class)):
    print("Precision for class", i, ":", precision_per_class[i])