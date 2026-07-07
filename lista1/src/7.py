import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import urllib.error

def read_url_pmm(url):
    try:
        # cabeçalho simples pra evitar bloqueios por user-agent
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as f:
            def read_word():
                word = b''
                while True:
                    c = f.read(1)
                    if not c:
                        break
                    if c == b'#':
                        f.readline()
                        continue
                    if c.isspace():
                        if word:
                            return word
                        continue
                    word += c
                return word

            magic = read_word().decode('ascii')
            width = int(read_word())
            height = int(read_word())
            maxval = int(read_word())

            if magic == 'P3':
                data = f.read().split()
                img = np.array(data, dtype=np.uint8).reshape((height, width, 3))
                return img
            else:
                raise ValueError(f"Infelizmente, o formato {magic} não é suportado. Este código suporta apenas P3 (ASCII)")
                
    except urllib.error.URLError as e:
        print(f"Erro ao acessar a URL: {e}")
        return None

def apply_filter(image, n, filter_matrix, save_path=None, return_float=False):        
    pad = n // 2
    kernel_flipped = np.flip(filter_matrix)
    img_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    h, w, c = image.shape
    output = np.zeros((h, w, c), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            output += img_padded[i:i+h, j:j+w, :] * kernel_flipped[i, j]
            
    if return_float:
        return output 
    else:
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

if __name__ == "__main__":
    url_imagem = "https://people.sc.fsu.edu/~jburkardt/data/ppma/blackbuck.ascii.ppm"
    
    img_rgb = read_url_pmm(url_imagem)
    
    if img_rgb is None:
        exit()

    img_gray = np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140])
    img_gray_3d = img_gray.reshape((img_gray.shape[0], img_gray.shape[1], 1))

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    n = 3
    gx_float = apply_filter(img_gray_3d, n, sobel_x, return_float=True)
    gy_float = apply_filter(img_gray_3d, n, sobel_y, return_float=True)

    magnitude = np.sqrt(gx_float**2 + gy_float**2)

    gx_view = np.clip(np.abs(gx_float), 0, 255).astype(np.uint8).squeeze()
    gy_view = np.clip(np.abs(gy_float), 0, 255).astype(np.uint8).squeeze()
    magnitude_view = np.clip(magnitude, 0, 255).astype(np.uint8).squeeze()
    original_view = img_gray.astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original_view, cmap='gray')
    axes[0, 0].set_title('original em tons de cinza')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gx_view, cmap='gray')
    axes[0, 1].set_title('arestas verticais - derivada horizontal (Gx)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gy_view, cmap='gray')
    axes[1, 0].set_title('arestas horizontais - derivada vertical (Gy)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(magnitude_view, cmap='gray')
    axes[1, 1].set_title('todas as arestas')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()