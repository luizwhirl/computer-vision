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

def apply_filter(image, n, filter_matrix):
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

    return np.clip(output, 0, 255).astype(np.uint8)

def gauss_filtro(k, sigma):
    limite = (k - 1) // 2
    vetor_coord = np.arange(-limite, limite + 1)
    x, y = np.meshgrid(vetor_coord, vetor_coord)

    termo_const = 1 / (2 * np.pi * sigma**2)
    expoente = -(x**2 + y**2) / (2 * sigma**2)

    H = termo_const * np.exp(expoente)
    return H / np.sum(H)

if __name__ == "__main__":
    img_rgb = read_ppm("caracol.ppm")
    img_cinza = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

    plt.imsave("caracol_cinza.ppm", img_cinza, cmap='gray')

    valores_k = [5, 15, 25]
    valores_sigma = [1.0, 2.5, 4.0]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i, k in enumerate(valores_k):
        for j, sigma in enumerate(valores_sigma):
            filtro = gauss_filtro(k, sigma)

            img_suavizada = apply_filter(img_cinza, k, filtro)

            ax = axes[i, j]
            ax.imshow(img_suavizada, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'k={k}, $\sigma$={sigma}')
            ax.axis('off')

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()