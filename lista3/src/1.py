import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gdown

# Caminhos locais dinâmicos
BASE_DIR = Path(__file__).parent.parent / "imagens"
OUT_DIR = Path(__file__).parent.parent / "4_results"

# Link da pasta do Google Drive com as imagens (retirado do Colab)
URL_DRIVE = "https://drive.google.com/drive/folders/1gwDOKvhtPspWmflDCIzU6NwHS43R3UsZ?usp=sharing"

TRIOS = {
    "livro":  ["livro1.jpg", "livro2.jpg", "livro3.jpg"],
    "caixa":  ["caixa1.jpg", "caixa2.jpg", "caixa3.jpg"],
    "hp":     ["hp1.jpg", "hp2.jpg", "hp3.jpg"],
    "lambda": ["lambda1.jpg", "lambda2.jpg", "lambda3.jpg"],
    "xero":   ["xero1.jpg", "xero2.jpg", "xero3.jpg"],
}

def baixar_imagens_drive():
    """Baixa a pasta de imagens do Google Drive se ainda não existir localmente."""
    if not BASE_DIR.exists() or len(list(BASE_DIR.rglob("*.jpg"))) == 0:
        print("A descarregar as imagens do Google Drive...")
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        # O gdown fará o download da pasta inteira
        gdown.download_folder(URL_DRIVE, output=str(BASE_DIR), quiet=False, use_cookies=False)
    else:
        print("Imagens já encontradas localmente. A ignorar o download.")

def carrega(caminho, max_lado=1100):
    img = cv2.imread(str(caminho))
    if img is None:
        raise FileNotFoundError(f"Não foi possivel ler a imagem em {caminho}")

    h, w = img.shape[:2]
    escala = min(1.0, max_lado / max(h, w))

    if escala < 1.0:
        img = cv2.resize(
            img,
            (int(w * escala), int(h * escala)),
            interpolation=cv2.INTER_AREA
        )
    return img

def pega_siftmatches(img1, img2, ratio=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=5000)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    pares = bf.knnMatch(des1, des2, k=2)

    bons_matches = []
    for par in pares:
        if len(par) != 2:
            continue
        m, n = par
        if m.distance < ratio * n.distance:
            bons_matches.append(m)

    return kp1, kp2, bons_matches

def calculahomografia(img_origem, img_destino):
    melhor_resultado = None

    for ratio in [0.75, 0.80, 0.85, 0.90]:
        kp1, kp2, matches = pega_siftmatches(img_origem, img_destino, ratio)

        if len(matches) < 4:
            continue

        pts_origem = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_destino = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mascara = cv2.findHomography(pts_origem, pts_destino, cv2.RANSAC, 5.0)

        if H is None:
            continue

        inliers = int(mascara.sum()) if mascara is not None else 0

        if melhor_resultado is None or inliers > melhor_resultado["inliers"]:
            melhor_resultado = {
                "H": H,
                "matches": len(matches),
                "inliers": inliers,
                "ratio": ratio,
            }

        if inliers >= 8:
            break

    if melhor_resultado is None:
        raise RuntimeError("Não foi possível calcular a homografia.")

    return melhor_resultado

def transformacoesPplano(H01, H12, plano):
    I = np.eye(3, dtype=np.float64)
    H10 = np.linalg.inv(H01)
    H21 = np.linalg.inv(H12)

    if plano == 1:
        return [I, H10, H10 @ H21]
    if plano == 2:
        return [H01, I, H21]
    if plano == 3:
        return [H12 @ H01, H12, I]

    raise ValueError("O plano deve ser 1, 2 ou 3")

def mescla_imagens(imagens, transformacoes, largura, altura):
    acumulado = np.zeros((altura, largura, 3), dtype=np.float32)
    pesos = np.zeros((altura, largura, 1), dtype=np.float32)

    for img, H in zip(imagens, transformacoes):
        warp = cv2.warpPerspective(img, H, (largura, altura))

        mascara = np.ones(img.shape[:2], dtype=np.uint8) * 255
        mascara_warp = cv2.warpPerspective(mascara, H, (largura, altura))
        mascara_warp = (mascara_warp > 0).astype(np.float32)[..., None]

        acumulado += warp.astype(np.float32) * mascara_warp
        pesos += mascara_warp

    panorama = acumulado / np.maximum(pesos, 1.0)
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    mascara_final = (pesos[..., 0] > 0).astype(np.uint8)
    coords = cv2.findNonZero(mascara_final)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        panorama = panorama[y:y + h, x:x + w]

    return panorama

def gera_panoramas(nome_trio, arquivos):
    imagens = []

    for arquivo in arquivos:
        caminhos_encontrados = list(BASE_DIR.rglob(arquivo))
        if not caminhos_encontrados:
             raise FileNotFoundError(f"O ficheiro {arquivo} não foi encontrado na pasta descarregada!")
        imagens.append(carrega(caminhos_encontrados[0]))

    resultado_01 = calculahomografia(imagens[0], imagens[1])
    resultado_12 = calculahomografia(imagens[1], imagens[2])

    H01 = resultado_01["H"]
    H12 = resultado_12["H"]

    print(f"\n[{nome_trio.upper()}] a processar homografias...")

    panoramas_gerados = []

    for plano in [1, 2, 3]:
        transformacoes = transformacoesPplano(H01, H12, plano)

        img_base = imagens[plano - 1]
        altura, largura = img_base.shape[:2]
        translacao = np.eye(3, dtype=np.float64)

        transformacoes_canvas = [translacao @ H for H in transformacoes]

        panorama = mescla_imagens(imagens, transformacoes_canvas, largura, altura)
        panoramas_gerados.append(panorama)

        saida = OUT_DIR / f"{nome_trio}_plano_imagem_{plano}.png"
        cv2.imwrite(str(saida), panorama)

    # Gráfico igual ao Colab
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(f"Trio: {nome_trio.upper()}", fontsize=18, fontweight='bold')

    titulos_originais = ["imagem 1", "imagem 2", "imagem 3"]
    titulos_panoramas = ["plano 1", "plano 2", "plano 3"]

    for i in range(3):
        ax = axes[0, i]
        img_rgb = cv2.cvtColor(imagens[i], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(titulos_originais[i], fontsize=12)
        ax.axis('off')

    for i in range(3):
        ax = axes[1, i]
        pano_rgb = cv2.cvtColor(panoramas_gerados[i], cv2.COLOR_BGR2RGB)
        ax.imshow(pano_rgb)
        ax.set_title(titulos_panoramas[i], fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print("-" * 60)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1º Passo: Garantir que as imagens existem chamando a função de download
    baixar_imagens_drive()
    
    print("Processamento iniciado (Metodologia: SIFT + BFMatcher kNN + teste de razão de Lowe + RANSAC)")

    # 2º Passo: Iterar sobre os trios de imagens
    for nome_trio, arquivos in TRIOS.items():
        try:
            gera_panoramas(nome_trio, arquivos)
        except Exception as e:
            print(f"Erro ao processar trio {nome_trio}: {e}")

    print(f"Panoramas salvos com sucesso em {OUT_DIR}")

if __name__ == "__main__":
    main()