import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

url = 'https://m.media-amazon.com/images/M/MV5BZThmYzBiYjItYThmYS00ZTEyLTg5YWQtNDQ4M2RlMTFjOGU4XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req)
arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def build_pyrmd(niveis):
    # pega as dimensões da imagem original
    h_orig, w_orig, c = niveis[0].shape
    
    canvas_h = h_orig
    canvas_w = w_orig + w_orig // 2
    canvas = np.zeros((canvas_h, canvas_w, c), dtype=np.uint8)
    
    # cola a original
    canvas[0:h_orig, 0:w_orig] = niveis[0]
    
    # cola as menores
    y_offset = 0
    x_offset = w_orig
    
    for i in range(1, len(niveis)):
        h_niv, w_niv, _ = niveis[i].shape
        # colando no deslocamento correto
        canvas[y_offset : y_offset + h_niv, x_offset : x_offset + w_niv] = niveis[i]
        y_offset += h_niv
        
    return canvas

pyrmd_ingenua = [img_rgb]
img_atual = img_rgb
for _ in range(3):
    # pega 1 pixel e pula o próximo
    img_atual = img_atual[::2, ::2]
    pyrmd_ingenua.append(img_atual)

pyrmd_gauss = [img_rgb]
img_atual = img_rgb
for _ in range(3):
    # aplica o blur gaussiano e DEPOIS reduz pela metade
    img_atual = cv2.pyrDown(img_atual)
    pyrmd_gauss.append(img_atual)

final_ingenua = build_pyrmd(pyrmd_ingenua)
final_gauss = build_pyrmd(pyrmd_gauss)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(final_ingenua)
axes[0].set_title('downsampling direto - Aliasing')
axes[0].axis('off')

axes[1].imshow(final_gauss)
axes[1].set_title('pirâmide gaussiana - Suavizada')
axes[1].axis('off')

plt.tight_layout()
plt.show()