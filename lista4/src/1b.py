import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import kagglehub

# Configurações globais
print("A descarregar a base de dados via kagglehub...")
dataset_path = kagglehub.dataset_download("tongpython/cat-and-dog")
class_names = ['cats', 'dogs']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

LIMITE_POR_CLASSE = 500  

filepaths = []
labels = []

print(f"A selecionar um limite de {LIMITE_POR_CLASSE} imagens por classe...")
for class_name in class_names:
    paths = glob.glob(os.path.join(dataset_path, '**', class_name, '*.*'), recursive=True)
    
    valid_count = 0
    for path in paths:
        # Filtro de segurança: ignora ficheiros vazios ou Thumbs.db
        if os.path.getsize(path) > 0 and path.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepaths.append(path)
            labels.append(class_name)
            valid_count += 1
            
        if valid_count >= LIMITE_POR_CLASSE:
            break

# Criamos uma tabela (DataFrame) com os caminhos que escolhemos
df = pd.DataFrame({'filename': filepaths, 'class': labels})
print(f"Total de imagens que serão utilizadas: {len(df)}")

# 1. Preparação dos Dados
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.30)

print("\nA configurar o carregamento das imagens de Treino (70%)...")
train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

print("A configurar o carregamento das imagens de Teste (30%)...")
test_gen = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# 2. Função para criar, treinar e avaliar as CNNs
def train_and_evaluate_cnn(base_model_class, model_name):
    print(f"\n" + "="*50)
    print(f" A iniciar o Treino do Modelo: {model_name}")
    print("="*50)
    
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, epochs=3, validation_data=test_gen)
    
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia Final {model_name}: {acc * 100:.2f}%")
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_gen.class_indices.keys()))
    disp.plot(cmap=plt.cm.Oranges)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()
    
    return model, y_true, y_pred

# 3. Execução do pipeline
model_vgg, true_vgg, pred_vgg = train_and_evaluate_cnn(VGG16, "VGG16")
model_res, true_res, pred_res = train_and_evaluate_cnn(ResNet50, "ResNet50")
model_mob, true_mob, pred_mob = train_and_evaluate_cnn(MobileNetV2, "MobileNetV2")

# 4. Exibição de Exemplos
filenames = test_gen.filenames
class_labels = list(test_gen.class_indices.keys())

acertos_idx = np.where(true_mob == pred_mob)[0]
erros_idx = np.where(true_mob != pred_mob)[0]

def plot_validation_examples(indices_list, title_text):
    if len(indices_list) == 0:
        print(f"Nenhum exemplo encontrado para: {title_text}")
        return
    fig, axes = plt.subplots(1, min(4, len(indices_list)), figsize=(15, 5))
    for i, idx in enumerate(indices_list[:4]):
        img_path = filenames[idx] 
        img = plt.imread(img_path)
        
        if len(img.shape) == 2: 
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
            
        axes[i].set_title(f"Real: {class_labels[true_mob[idx]]}\nPred: {class_labels[pred_mob[idx]]}")
        axes[i].axis('off')
    fig.suptitle(title_text, fontsize=16)
    plt.show()

plot_validation_examples(acertos_idx, "Exemplos de ACERTOS - MobileNetV2")
plot_validation_examples(erros_idx, "Exemplos de ERROS - MobileNetV2")