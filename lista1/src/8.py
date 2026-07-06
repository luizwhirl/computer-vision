import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

url = 'https://www.themarysue.com/wp-content/uploads/2015/09/Glados.jpg?resize=1024%2C768'

# requisição para pegar a imagem
requisicao = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
)
resposta = urllib.request.urlopen(requisicao)
arr = np.asarray(bytearray(resposta.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny1 = cv2.Canny(img_gray, threshold1=100, threshold2=200, apertureSize=3) # 1- padrão/equilibrado
canny2 = cv2.Canny(img_gray, threshold1=30, threshold2=80, apertureSize=3) # 2- limiares baixos: muito sensíveis, detecta até ruidos
canny3 = cv2.Canny(img_gray, threshold1=200, threshold2=250, apertureSize=3) # 3- limiares altos: pouco sensível, só bordas fortes
canny4 = cv2.Canny(img_gray, threshold1=1000, threshold2=2500, apertureSize=5) # 4- com apertureSize=5 os valores ficam tão maiores que o threeshold precisa ser maior para nao ficar tudo preto

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(canny1, cmap='gray')
axes[0, 0].set_title('1. padrão (T1=100, T2=200, Ap=3)')
axes[0, 0].axis('off')

axes[0, 1].imshow(canny2, cmap='gray')
axes[0, 1].set_title('2. sensível (T1=30, T2=80, Ap=3)')
axes[0, 1].axis('off')

axes[1, 0].imshow(canny3, cmap='gray')
axes[1, 0].set_title('3. rigoroso (T1=200, T2=250, Ap=3)')
axes[1, 0].axis('off')

axes[1, 1].imshow(canny4, cmap='gray')
axes[1, 1].set_title('4. kernel maior (T1=1000, T2=2500, Ap=5)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()