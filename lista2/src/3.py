import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def carrega_extrai(diretorio, descritor):
    features_dict = {}
    arquivos = [f for f in os.listdir(diretorio) if f.endswith(('.png', '.jpg'))]
    
    for arquivo in arquivos:
        caminho = os.path.join(diretorio, arquivo)
        img = cv2.imread(caminho)
        if img is None:
            print(f"Erro ao ler imagem: {caminho}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = descritor.detectAndCompute(gray, None)
        features_dict[arquivo] = {'keypoints': kp, 'descriptors': des, 'image': img}
        
    return features_dict

def identifica():
    labels_dir = "imagens-3/labels"
    products_dir = "imagens-3/products"
    
    # esse dicionario mapeia cada foto de produto capturada para o rótulo correspondente na base de rótulos (ground truth)
    # coluna esq: produtos
    # coluna dir: rótulos (ground truth)
    ground_truth = {
        "Pimage1.jpg": "image1.jpg",
        "Pimage2.jpg": "image9.jpg",
        "Pimage3.jpg": "image11.jpg",
        "Pimage4.jpg": "image2.jpg",
        "Pimage5.jpg": "image10.jpg",
        "Pimage6.jpg": "image6.jpg",
        "Pimage7.jpg": "image8.jpg", 
    }

    # inicia o SIFT  
    sift = cv2.SIFT_create()
    
    # carrega a base de rótulos
    print("Processando base de rótulos...")
    labels_features = carrega_extrai(labels_dir, sift)
    
    print("\nProcessando fotos dos produtos...")
    produtos_features = carrega_extrai(products_dir, sift)
    
    # config BFMatcher
    bf = cv2.BFMatcher()

    y_true = []
    y_pred = []
    
    # percorre cada foto produto 
    for prod_name, prod_data in produtos_features.items():
        if prod_name not in ground_truth:
            continue
            
        des_prod = prod_data['descriptors']
        
        melhor_label = "Desconhecido"
        max_bons_matches = -1
        
        # compara com TODOS rotulos da base
        for label_name, label_data in labels_features.items():
            des_label = label_data['descriptors']
            
            # se-> não tem descritores suficientes, pula
            if des_prod is None or des_label is None or len(des_prod) < 2 or len(des_label) < 2:
                continue
                
            # KNN Match com k=2 para aplicar o Teste de Lowe
            matches = bf.knnMatch(des_prod, des_label, k=2)
            
            # aplica o Lowe's  Ratio Test
            bons_matches = 0
            for match_points in matches:
                if len(match_points) == 2:
                    m, n = match_points
                    if m.distance < 0.75 * n.distance:
                        bons_matches += 1
            
            # se encontrou um rótulo com mais correspondências -> atualiza
            if bons_matches > max_bons_matches:
                max_bons_matches = bons_matches
                melhor_label = label_name
                
        print(f"Produto: {prod_name} | Predição: {melhor_label} | Matches: {max_bons_matches}")
        
        # guarda resultados para as metricas
        y_true.append(ground_truth[prod_name])
        y_pred.append(melhor_label)

    # d)Acurácia 
    acuracia = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia do Sistema: {acuracia * 100:.2f}%")

    # constroi a da matriz de confusão
    # lista unica de todos os rotulos possiveis para os eixos do grafico
    classes_unicas = sorted(list(labels_features.keys()))
    
    matriz_confusao = confusion_matrix(y_true, y_pred, labels=classes_unicas)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=classes_unicas)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title("Matriz de Confusão - Identificação de Produtos")
    plt.xlabel("Classe Predita (Algoritmo)")
    plt.ylabel("Classe Real (Ground Truth)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    identifica()