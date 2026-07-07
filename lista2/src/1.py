import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def aplicar_detectores(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Não foi possível carregar a imagem em {img_path}")
        return

    # converte para rgb
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_harris = img_rgb.copy()
    img_shitomasi = img_rgb.copy()
    img_fast = img_rgb.copy()

    # Harris
    # parâmetros: img, blockSize, ksize, k
    gray_float = np.float32(gray)
    dst_harris = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    
    # dilata o resultado para marcar os cantos mais visivelmente
    dst_harris = cv2.dilate(dst_harris, None)
    
    # exibir os cantos em ermelho
    img_harris[dst_harris > 0.01 * dst_harris.max()] = [255, 0, 0]

    # shi-tomasi
    # img, maxCorners, qualityLevel, minDistance
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.int32(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img_shitomasi, (x, y), 5, (0, 255, 0), -1)

    # FAST
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    
    # encontra e desenha os keypoints
    kp = fast.detect(gray, None)
    img_fast = cv2.drawKeypoints(img_fast, kp, None, color=(0, 0, 255))

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"{os.path.basename(img_path)}", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(img_harris)
    plt.title('Harris')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_shitomasi)
    plt.title('Shi-Tomasi')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_fast)
    plt.title('FAST')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for i in range(1, 8):
        caminho = f"imagens-1/image{i}.png" 
        print(f"Processando: {caminho}...")
        aplicar_detectores(caminho)