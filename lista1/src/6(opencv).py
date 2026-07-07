import cv2
import matplotlib.pyplot as plt

entrada_path = 'entradaq6.png'
img = cv2.imread(entrada_path)

if img is None:
    print(f"arquivo '{entrada_path}' foi encontrado")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_mediana_7 = cv2.medianBlur(img_rgb, 7)
    img_nlm = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_mediana_7)
    plt.title('filtro mediana (7x7)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_nlm)
    plt.title('non-local means')
    plt.axis('off')

    plt.tight_layout()
    plt.show()