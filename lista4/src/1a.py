import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import kagglehub

print("A descarregar a base de dados via kagglehub para evitar alojamento local...")
# Download gerido pela cache do Kaggle (como no Colab)
dataset_path = kagglehub.dataset_download("tongpython/cat-and-dog")

# Adaptado para as pastas usuais extraídas
class_names = ['cats', 'dogs']

def extract_hog_features(image_path):
    img = cv2.imread(image_path)
    # Proteção contra imagens corrompidas ou ficheiros não-imagem
    if img is None or img.size == 0:
        return None, None
    
    # redimensionando p/ HOG (64x128)
    img = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten(), img

X_hog, y_hog, images_hog = [], [], []

print("A extrair descritores HOG das imagens...")
for label, class_name in enumerate(class_names):
    # Procura recursiva de imagens ('**') independente da pasta root ser 'train' ou 'training_set'
    paths = glob.glob(os.path.join(dataset_path, '**', class_name, '*.*'), recursive=True)
    
    # limitando a 500 imagens por classe para agilizar o treino
    for path in paths[:500]: 
        features, img = extract_hog_features(path)
        if features is not None:
            X_hog.append(features)
            y_hog.append(label)
            images_hog.append(img)

X_hog = np.array(X_hog)
y_hog = np.array(y_hog)

print(f"Total de amostras carregadas com sucesso: {len(X_hog)}")

# 2. Divisão Holdout (70% Treino, 30% Teste)
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X_hog, y_hog, images_hog, test_size=0.30, random_state=42
)

# 3. Treinamento do SVM
print("A treinar o classificador SVM (pode demorar alguns segundos)...")
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)

# 4. Previsões e Métricas
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n=========================================")
print(f"Acurácia atingida pelo SVM + HOG: {acc * 100:.2f}%")
print(f"=========================================\n")

# 5. Exibição da Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - HOG + SVM (Cats vs Dogs)")
plt.show()

# 6. Exibir Exemplos de Classificação
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(cv2.cvtColor(img_test[i], cv2.COLOR_BGR2RGB))
    ax.set_title(f"Real: {class_names[y_test[i]]}\nPred: {class_names[y_pred[i]]}")
    ax.axis('off')
plt.suptitle("Exemplos de Classificação com HOG + SVM")
plt.show()