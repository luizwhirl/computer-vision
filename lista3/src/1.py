from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "4_results"

TRIOS = {
    "livro":  ["livro1.jpg", "livro2.jpg", "livro3.jpg"],
    "caixa":   ["caixa1.jpg", "caixa2.jpg", "caixa3.jpg"],
    "hp":  ["hp1.jpg", "hp2.jpg", "hp3.jpg"],
    "lambda": ["lambda1.jpg", "lambda2.jpg", "lambda3.jpg"],
    "xero":   ["xero1.jpg", "xero2.jpg", "xero3.jpg"],
}


def carregar_imagem(caminho, max_lado=1100):
    img = cv2.imread(str(caminho))
    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")

    h, w = img.shape[:2]
    escala = min(1.0, max_lado / max(h, w))

    if escala < 1.0:
        img = cv2.resize(
            img,
            (int(w * escala), int(h * escala)),
            interpolation=cv2.INTER_AREA
        )

    return img


def obter_matches_sift(img1, img2, ratio=0.75):
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


def calcular_homografia(img_origem, img_destino):
    melhor_resultado = None

    # Comeca com Lowe 0.75 e relaxa se houver poucas correspondencias boas.
    for ratio in [0.75, 0.80, 0.85, 0.90]:
        kp1, kp2, matches = obter_matches_sift(img_origem, img_destino, ratio)

        if len(matches) < 4:
            continue

        pts_origem = np.float32(
            [kp1[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        pts_destino = np.float32(
            [kp2[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        H, mascara = cv2.findHomography(
            pts_origem,
            pts_destino,
            cv2.RANSAC,
            5.0
        )

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
        raise RuntimeError("Nao foi possivel calcular a homografia.")

    return melhor_resultado


def calcular_canvas(imagens, transformacoes):
    todos_cantos = []

    for img, H in zip(imagens, transformacoes):
        h, w = img.shape[:2]

        cantos = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)

        cantos_transformados = cv2.perspectiveTransform(cantos, H)
        todos_cantos.append(cantos_transformados)

    todos_cantos = np.concatenate(todos_cantos, axis=0)

    xmin, ymin = np.floor(todos_cantos.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(todos_cantos.max(axis=0).ravel()).astype(int)

    largura = xmax - xmin
    altura = ymax - ymin

    translacao = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ], dtype=np.float64)

    return translacao, largura, altura


def mesclar_imagens(imagens, transformacoes, largura, altura):
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


def transformacoes_para_plano(H01, H12, plano):
    I = np.eye(3, dtype=np.float64)

    H10 = np.linalg.inv(H01)
    H21 = np.linalg.inv(H12)

    if plano == 1:
        # Alinhando tudo no plano da imagem 1
        return [I, H10, H10 @ H21]

    if plano == 2:
        # Alinhando tudo no plano da imagem 2
        return [H01, I, H21]

    if plano == 3:
        # Alinhando tudo no plano da imagem 3
        return [H12 @ H01, H12, I]

    raise ValueError("Plano deve ser 1, 2 ou 3.")


def gerar_panoramas(nome_trio, arquivos):
    imagens = [
        carregar_imagem(BASE_DIR / arquivo)
        for arquivo in arquivos
    ]

    resultado_01 = calcular_homografia(imagens[0], imagens[1])
    resultado_12 = calcular_homografia(imagens[1], imagens[2])

    H01 = resultado_01["H"]  # imagem 1 -> imagem 2
    H12 = resultado_12["H"]  # imagem 2 -> imagem 3

    print(f"\nTrio: {nome_trio}")
    print(
        f"Imagem 1 -> 2: {resultado_01['matches']} matches, "
        f"{resultado_01['inliers']} inliers, Lowe={resultado_01['ratio']}"
    )
    print(
        f"Imagem 2 -> 3: {resultado_12['matches']} matches, "
        f"{resultado_12['inliers']} inliers, Lowe={resultado_12['ratio']}"
    )

    for plano in [1, 2, 3]:
        transformacoes = transformacoes_para_plano(H01, H12, plano)

        # Utiliza as proporções exclusivas da imagem que está sendo usada como base
        img_base = imagens[plano - 1]
        altura, largura = img_base.shape[:2]
        translacao = np.eye(3, dtype=np.float64) # Mantém a matriz de translação neutra

        transformacoes_canvas = [
            translacao @ H
            for H in transformacoes
        ]

        panorama = mesclar_imagens(
            imagens,
            transformacoes_canvas,
            largura,
            altura
        )

        saida = OUT_DIR / f"{nome_trio}_plano_imagem_{plano}.png"
        cv2.imwrite(str(saida), panorama)

        print(f"Panorama salvo: {saida}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Metodologia escolhida:")
    print("SIFT + BFMatcher kNN + teste de razao de Lowe + RANSAC")
    print("Nao foi usada a API Stitcher.")

    for nome_trio, arquivos in TRIOS.items():
        gerar_panoramas(nome_trio, arquivos)

    print("\nFinalizado.")
    print(f"Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()