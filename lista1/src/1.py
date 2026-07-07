import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# URL da imagem original usada no Colab
URL_IMAGEM = "https://people.sc.fsu.edu/~jburkardt/data/ppma/snail.ascii.ppm"

def read_ppm(filepath):
    with open(filepath, 'rb') as f:
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

def write_ppm(image, filepath):
    h, w, c = image.shape
    with open(filepath, 'wb') as f:
        header = f"P6\n{w} {h}\n255\n".encode('ascii')
        f.write(header)
        f.write(image.tobytes())

def apply_filter(image, n, filter_matrix, save_path=None):        
    pad = n // 2
    kernel_flipped = np.flip(filter_matrix)
    img_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    h, w, c = image.shape
    output = np.zeros((h, w, c), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            output += img_padded[i:i+h, j:j+w, :] * kernel_flipped[i, j]
            
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    if save_path:
        write_ppm(output, save_path)        
    return output

if __name__ == "__main__":
    entrada = "caracol.ppm"
    saida = "filtrada.ppm"
    
    # Automatização: Verifica se a imagem já foi baixada. Se não, faz o download automático.
    if not os.path.exists(entrada):
        print(f"O ficheiro '{entrada}' não foi encontrado. Iniciando o download...")
        try:
            urllib.request.urlretrieve(URL_IMAGEM, entrada)
            print(f"Download concluído: '{entrada}' salvo com sucesso!")
        except Exception as e:
            print(f"Erro ao baixar a imagem: {e}")
            exit(1)
            
    imagem = read_ppm(entrada)
    
    n = 5
    pesosmatriz = np.ones((n, n)) / (n * n)
    resultado = apply_filter(imagem, n, pesosmatriz, save_path=saida)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(imagem)
    axes[0].set_title('original')
    axes[0].axis('off')
    
    axes[1].imshow(resultado)
    axes[1].set_title(f'filtro blur ({n}x{n})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()