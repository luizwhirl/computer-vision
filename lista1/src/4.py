import numpy as np
import matplotlib.pyplot as plt

def read_ppm(filepath):
    with open(filepath, 'rb') as f:
        def read_word():
            word = b''
            while True:
                c = f.read(1)
                if not c: break
                if c == b'#':
                    f.readline(); continue
                if c.isspace():
                    if word: return word
                    continue
                word += c
            return word

        magic = read_word().decode('ascii')
        width = int(read_word())
        height = int(read_word())
        maxval = int(read_word())

        if magic == 'P6':
            data = f.read()
            expected_size = width * height * 3
            img = np.frombuffer(data[:expected_size], dtype=np.uint8).reshape((height, width, 3))
        elif magic == 'P3':
            data = f.read().split()
            img = np.array(data, dtype=np.uint8).reshape((height, width, 3))
        return img

def apply_filter(image, n, filter_matrix, offset=0):
    pad = n // 2
    kernel_flipped = np.flip(filter_matrix)
    is_grayscale = (len(image.shape) == 2)

    if is_grayscale:
        img_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
        h, w = image.shape
        output = np.zeros((h, w), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                output += img_padded[i:i+h, j:j+w] * kernel_flipped[i, j]
    else:
        img_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        h, w, c = image.shape
        output = np.zeros((h, w, c), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                output += img_padded[i:i+h, j:j+w, :] * kernel_flipped[i, j]

    output = output + offset
    return np.clip(output, 0, 255).astype(np.uint8)

def gauss_filtro(k, sigma):
    limite = (k - 1) // 2
    vetor_coord = np.arange(-limite, limite + 1)
    x, y = np.meshgrid(vetor_coord, vetor_coord)
    termo_const = 1 / (2 * np.pi * sigma**2)
    expoente = -(x**2 + y**2) / (2 * sigma**2)
    H = termo_const * np.exp(expoente)
    return H / np.sum(H)

def passalta_filtro(k, sigma):
    filtro_baixa = gauss_filtro(k, sigma)

    filtro_identidade = np.zeros((k, k))
    centro = k // 2
    filtro_identidade[centro, centro] = 1.0

    filtro_alta = filtro_identidade - filtro_baixa
    return filtro_alta

# 4.
if __name__ == "__main__":
    # 4.1
    img_lida = read_ppm("caracol_cinza.ppm")

    if len(img_lida.shape) == 3:
        img_cinza = img_lida[:, :, 0]
    else:
        img_cinza = img_lida

    # 4.2
    combinacoes = [
        (5, 1.0),
        (15, 2.5),
        (25, 4.0)
    ]

    # 4.3
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(img_cinza, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('orignal')
    axes[0].axis('off')

    for idx, (k, sigma) in enumerate(combinacoes):
        filtro_hp = passalta_filtro(k, sigma)
        img_alta = apply_filter(img_cinza, k, filtro_hp, offset=128)
        axes[idx + 1].imshow(img_alta, cmap='gray', vmin=0, vmax=255)

        axes[idx + 1].set_title(f'passa-alta (k={k}, $\sigma$={sigma})')
        axes[idx + 1].axis('off')

    plt.tight_layout()
    plt.show()