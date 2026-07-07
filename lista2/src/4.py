import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

def detecao_blobs_log(image, num_escalas=6, sigma_0=1.0, k=np.sqrt(2), threshold=0.03):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray_norm = gray.astype(np.float64) / 255.0

    h, w = gray_norm.shape
    
    # matriz (x, y, s)
    scale_space = np.zeros((h, w, num_escalas))
    sigmas = np.zeros(num_escalas)

    # filtro LoG
    for s in range(num_escalas):
        # sigma_s = sigma_0 * (k^s)
        sigma = sigma_0 * (k ** s)
        sigmas[s] = sigma

        ksize = int(2 * np.ceil(3 * sigma) + 1)
        blur = cv2.GaussianBlur(gray_norm, (ksize, ksize), sigma)
        # filtro laplaciano
        laplacian = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
        # multiplica resposta por sigma^2
        scale_space[:, :, s] = np.abs(laplacian) * (sigma ** 2)

    # busca vizinhança 3x3x3 - maximos locais
    is_max = np.ones((h, w, num_escalas), dtype=bool)

    # comparação com np.roll dos26 vizinhos
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            for ds in [-1, 0, 1]:
                if dy == 0 and dx == 0 and ds == 0:
                    continue
                shifted = np.roll(scale_space, shift=(dy, dx, ds), axis=(0, 1, 2))
                is_max &= (scale_space > shifted)

    # ecluindo s=0 e s=5
    is_max[:, :, 0] = False
    is_max[:, :, -1] = False

    # excluindo pixels da borda da imagem
    is_max[0, :, :] = False
    is_max[-1, :, :] = False
    is_max[:, 0, :] = False
    is_max[:, -1, :] = False

    is_max &= (scale_space > threshold)
    y_coords, x_coords, s_coords = np.where(is_max)

    # output_img = image.copy()
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # cinza -> converte para BGR (para os blobs)
    if len(output_img.shape) == 2:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

    for y, x, s in zip(y_coords, x_coords, s_coords):
        sigma = sigmas[s]
        # raio blob r = sqrt(2) * sigma
        r = int(np.sqrt(2) * sigma)
        
        # blob verde (x, y) e raio r
        cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)

    return output_img

if __name__ == "__main__":
    url = "https://images.myglobalflowers.com/aff31852-7d06-4979-9d85-d4ef6e535200/original"
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img_teste = cv2.imdecode(arr, -1)

    # lembrete pra antes de enviar: ajustar thresold
    img_resultado = detecao_blobs_log(img_teste, num_escalas=6, sigma_0=1.0, threshold=0.35)
    plt.figure(figsize=(12, 12))
    # convertendo BGR -> RGB
    plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('detecção de blobs com LoG')
    plt.show()