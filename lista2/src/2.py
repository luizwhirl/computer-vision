import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def processa_par(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None:
        print(f"Não foi possível carregar {img_path1}")
        return
    if img2 is None:
        print(f"Não foi possível carregar {img_path2}")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def extrai_combina(descritor, norm_type, nome_descritor):
        kp1, des1 = descritor.detectAndCompute(gray1, None)
        kp2, des2 = descritor.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            return np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

        bf = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:30], None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # 1. SIFT
    sift = cv2.SIFT_create()
    img_sift = extrai_combina(sift, cv2.NORM_L2, "SIFT")

    # 2. ORB
    orb = cv2.ORB_create(nfeatures=1000)
    img_orb = extrai_combina(orb, cv2.NORM_HAMMING, "ORB")

    # 3. BRISK (agora vai funcionar nativamente!)
    brisk = cv2.BRISK_create()
    img_brisk = extrai_combina(brisk, cv2.NORM_HAMMING, "BRISK")

    plt.figure(figsize=(18, 6))
    nome = os.path.basename(img_path1)
    nome_par = os.path.basename(img_path2)
    plt.suptitle(f"{nome} vs {nome_par}", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(img_sift)
    plt.title('SIFT Matching - NORM_L2')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_orb)
    plt.title('ORB Matching - NORM_HAMMING')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_brisk)
    plt.title('BRISK Matching - NORM_HAMMING')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Garante que as imagens são procuradas a partir da pasta correta
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    pasta_imagens = os.path.join(diretorio_atual, "imagens-2")

    for i in range(1, 5):
        img_original = os.path.join(pasta_imagens, f"image{i}.png")
        img_par = os.path.join(pasta_imagens, f"image{i}-par.png")
        
        print(f"Processando: {img_original} e {img_par}...")
        processa_par(img_original, img_par)
        
    print("Processamento concluído")