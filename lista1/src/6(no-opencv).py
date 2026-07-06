import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
import os

def filtro_bilateral(imagem, k, sigma_espacial, sigma_cor):
    pad = k // 2
    
    is_grayscale = imagem.ndim == 2
    if is_grayscale:
        imagem = imagem[:, :, np.newaxis]
        
    img_float = imagem.astype(np.float32)
    img_padded = np.pad(img_float, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    janelas = sliding_window_view(img_padded, window_shape=(k, k), axis=(0, 1))
    janelas = np.moveaxis(janelas, 2, -1)
    
    y, x = np.ogrid[-pad:pad+1, -pad:pad+1]
    kernel_espacial = np.exp(-(x**2 + y**2) / (2 * sigma_espacial**2))
    
    centros = img_float[:, :, np.newaxis, np.newaxis, :]
    diff = janelas - centros
    diff_squared = np.sum(diff**2, axis=-1)
    
    kernel_cor = np.exp(-(diff_squared) / (2 * sigma_cor**2))
    
    pesos_finais = kernel_espacial * kernel_cor
    
    soma_pondr = np.sum(janelas * pesos_finais[..., np.newaxis], axis=(2, 3))
    soma_pesos = np.sum(pesos_finais, axis=(2, 3))
    
    img_filtrada = soma_pondr / soma_pesos[..., np.newaxis]
    
    if is_grayscale:
        img_filtrada = img_filtrada.squeeze(axis=-1)
        
    return np.clip(img_filtrada, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    entrada_path = "entradaq6.png"
    
    if not os.path.exists(entrada_path):
        print(f"arquivo '{entrada_path}' não foi encontrado.")
    else:
        img_pil = Image.open(entrada_path).convert('RGB')
        entrada = np.array(img_pil)
        
        # k = 9
        # sigma_s = 3.0
        # sigma_c = 40.0
        
        k = 11
        sigma_s = 5.0
        sigma_c = 80.0

        img_limpa = filtro_bilateral(entrada, k, sigma_s, sigma_c)
        
        is_gray = entrada.ndim == 2
        cmap = 'gray' if is_gray else None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(entrada, cmap=cmap)
        axes[0].set_title('original')
        axes[0].axis('off')
        
        axes[1].imshow(img_limpa, cmap=cmap)
        titulo = f'resultado - bilateral k={k}, $\sigma_s$={sigma_s}, $\sigma_c$={sigma_c:.1f}'
        axes[1].set_title(titulo)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()