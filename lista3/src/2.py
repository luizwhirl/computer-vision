import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('soccer.jpg')
print(f"tamanho da imagem: {img.shape[1]}x{img.shape[0]} pixels")
pts_src = np.array([
    [862,  1600],
    [2592, 4010],
    [4162, 1190],
    [6727, 2985]
], dtype=float) 

escala = 10
largura_img = int(105*escala)
altura_img  = int(68*escala)

pts_dst_img = np.array([
    [0,0],
    [0,altura_img ],
    [largura_img, 0],
    [largura_img, altura_img]
], dtype=float)

pts_dst_metros = np.array([
    [0,68],
    [0,0],
    [105,68],
    [105,0]
], dtype=float)

Himg,_ = cv2.findHomography(pts_src, pts_dst_img)
Hmetros,_ = cv2.findHomography(pts_src, pts_dst_metros)
img_out = cv2.warpPerspective(img, Himg, (largura_img, altura_img))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask1 = cv2.inRange(hsv, np.array([0,   120, 70]), np.array([10,  255, 255]))
mask2 = cv2.inRange(hsv, np.array([160, 120, 70]), np.array([180, 255, 255]))
mask_red = cv2.bitwise_or(mask1, mask2)

kernel = np.ones((9, 9), np.uint8)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,  kernel, iterations=2)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#(min_area / max_area)
min_area = 300
max_area = 80_000

jogadores = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if min_area < area < max_area:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            jogadores.append((cx, cy, area))

# ordena do: maior blob para menor
jogadores.sort(key=lambda x: x[2], reverse=True)
print(f"\ntotal de blobs detectados: {len(jogadores)}")

def projeta_ponto(H, ponto_pixel):
    p = np.array([ponto_pixel[0], ponto_pixel[1], 1.0])
    p_t = H @ p
    return p_t[0] / p_t[2], p_t[1] / p_t[2]

# desenhamento de blobss
cores_bgr = [
    (255,   0, 255),
]

print("\n coordenadas canto inferior esquerdo (metros) = 0,0) ")
validos = 0
for i, (cx, cy, area) in enumerate(jogadores):
    x_img, y_img = projeta_ponto(Himg,    (cx, cy))
    x_m,   y_m   = projeta_ponto(Hmetros, (cx, cy))

    # if -> ponto cair dentro do campo projetado
    # aí desenha
    if 0 <= x_img <= largura_img and 0 <= y_img <= altura_img:
        cor = cores_bgr[validos % len(cores_bgr)]
        cv2.circle(img_out, (int(x_img), int(y_img)), 12, cor, -1)
        cv2.putText(img_out, str(validos + 1),
                    (int(x_img) + 16, int(y_img) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
        print(f"  jogador {validos+1}: pixel_orig=({cx},{cy})  →  "
              f"X = {x_m:.2f} m,  Y = {y_m:.2f} m  (área={area:.0f} px²)")
        validos += 1

img_rgb_orig = cv2.cvtColor(img,     cv2.COLOR_BGR2RGB)
img_rgb_out  = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

axes[0].imshow(img_rgb_orig)
axes[0].set_title('imagem original', fontsize=13)
axes[0].axis('off')

axes[1].imshow(mask_red, cmap='gray')
axes[1].set_title('detecção', fontsize=13)
axes[1].axis('off')

axes[2].imshow(img_rgb_out)
axes[2].set_title('vista superior', fontsize=13)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('resultado_homografia.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nimagem salva")