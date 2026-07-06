import numpy as np

def cria_filtro(k, sigma):
    if k % 2 == 0:
        raise ValueError("o tamanho do filtro k tem q ser um número ímpar")
        
    # 1
    limite = (k - 1) // 2
    
    vetor_coord = np.arange(-limite, limite + 1)
    
    x, y = np.meshgrid(vetor_coord, vetor_coord)
    
    # 2 e 3
    const = 1 / (2 * np.pi * sigma**2)
    expoente = -(x**2 + y**2) / (2 * sigma**2)
    
    H = const * np.exp(expoente)
    
    # 4
    H_normalizado = H / np.sum(H)
    
    return H_normalizado

if __name__ == "__main__":
    tamanho_k = 5
    desvio_sigma = 1.0
    
    maskara = cria_filtro(tamanho_k, desvio_sigma)
    
    print(f"filtro gausiano {tamanho_k}x{tamanho_k} (sigma={desvio_sigma}):\n")
    print(np.round(maskara, 4))
    print(f"\nsoma dos elementos (esperado 1.0): {np.sum(maskara)}")