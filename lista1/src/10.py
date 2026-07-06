import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

url = 'https://media.moddb.com/images/members/5/4611/4610303/profile/TLL.jpg'
# url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIUbHuA3IQPnXhrFPyDVrc20oM7vYqtck5eg&s'
# url = 'https://hips.hearstapps.com/hmg-prod/images/white-cat-breeds-kitten-in-grass-67bf648a54a3b.jpg?crop=0.668xw:1.00xh;0.167xw,0&resize=1200:*'
# url = 'https://static.wikia.nocookie.net/chespirito/images/3/39/Chaves7517_480.jpg/revision/latest/scale-to-width-down/258?cb=20180420232123&path-prefix=pt'

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req)

image = np.asarray(bytearray(resp.read()), dtype="uint8")
img = cv2.imdecode(image, cv2.IMREAD_COLOR)
# bgr -> rgb
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# quadriplicando as dimensões originais
altura_orig, largura_orig = img_rgb.shape[:2]
nova_largura = largura_orig * 4
nova_altura = altura_orig * 4

# redimensionamento bilinear
img_linear = cv2.resize(img_rgb, (nova_largura, nova_altura), interpolation=cv2.INTER_LINEAR)
img_bicubica = cv2.resize(img_rgb, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img_rgb)
ax1.set_title(f"original\n({largura_orig}x{altura_orig})")
ax1.axis('off')

ax2.imshow(img_linear)
ax2.set_title(f"interpolação linear\n({nova_largura}x{nova_altura})")
ax2.axis('off')

ax3.imshow(img_bicubica)
ax3.set_title(f"interpolação bicúbica\n({nova_largura}x{nova_altura})")
ax3.axis('off')

plt.tight_layout()
plt.show()

"""
crop_y, crop_x = nova_altura//2, nova_largura//2
tamanho_crop = 200

fig_zoom, (az1, az2) = plt.subplots(1, 2, figsize=(12, 6))
az1.imshow(img_linear[crop_y:crop_y+tamanho_crop, crop_x:crop_x+tamanho_crop])
az1.set_title("Zoom - Linear")
az2.imshow(img_bicubica[crop_y:crop_y+tamanho_crop, crop_x:crop_x+tamanho_crop])
az2.set_title("Zoom - Bicúbica")
plt.show()
"""