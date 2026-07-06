from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "4-results"

PAIRS = [
    ("garrafa", "garrafa_esq.png", "garrafa_dir.png"),
    ("livro", "livro2.jpg", "livro3.jpg"),
    ("gar", "gar2.jpg", "gar3.jpg"),
]


def load_image(filename, max_side=950):
    path = BASE_DIR / filename
    img = cv2.imread(str(path))

    if img is None:
        raise FileNotFoundError(f"Nao consegui abrir: {path}")

    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))

    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return img


def to_gray_equalized(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def detect_matches(img_l, img_r):
    gray_l = to_gray_equalized(img_l)
    gray_r = to_gray_equalized(img_r)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=6000)
        norm = cv2.NORM_L2
        ratio = 0.74
    else:
        detector = cv2.ORB_create(nfeatures=8000)
        norm = cv2.NORM_HAMMING
        ratio = 0.78

    kp_l, des_l = detector.detectAndCompute(gray_l, None)
    kp_r, des_r = detector.detectAndCompute(gray_r, None)

    if des_l is None or des_r is None:
        raise RuntimeError("Nao foram encontrados pontos suficientes.")

    matcher = cv2.BFMatcher(norm)
    raw_matches = matcher.knnMatch(des_l, des_r, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 10:
        raise RuntimeError("Poucos matches bons entre as imagens.")

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])

    return pts_l, pts_r, len(good)


def rectify_uncalibrated(img_l, img_r):
    h, w = img_l.shape[:2]

    pts_l, pts_r, total_matches = detect_matches(img_l, img_r)

    F, inlier_mask = cv2.findFundamentalMat(
        pts_l,
        pts_r,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.2,
        confidence=0.995,
        maxIters=5000
    )

    if F is None or inlier_mask is None:
        raise RuntimeError("Nao foi possivel estimar a matriz fundamental.")

    inlier_mask = inlier_mask.ravel().astype(bool)
    pts_l_in = pts_l[inlier_mask]
    pts_r_in = pts_r[inlier_mask]

    if len(pts_l_in) < 10:
        raise RuntimeError("Poucos inliers para retificar o par.")

    ok, H_l, H_r = cv2.stereoRectifyUncalibrated(
        pts_l_in,
        pts_r_in,
        F,
        imgSize=(w, h)
    )

    if not ok:
        raise RuntimeError("A retificacao nao convergiu.")

    left_rect = cv2.warpPerspective(img_l, H_l, (w, h))
    right_rect = cv2.warpPerspective(img_r, H_r, (w, h))

    print(f"  matches bons: {total_matches} | inliers: {len(pts_l_in)}")

    return left_rect, right_rect


def compute_disparity(img_l, img_r):
    gray_l = to_gray_equalized(img_l)
    gray_r = to_gray_equalized(img_r)

    h, w = gray_l.shape

    block_size = 5
    min_disp = 0

    num_disp = max(16 * 6, int(w / 8))
    num_disp = ((num_disp // 16) + 1) * 16
    num_disp = min(num_disp, 16 * 12)

    stereo_l = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=7,
        speckleWindowSize=120,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    stereo_r = cv2.StereoSGBM_create(
        minDisparity=-num_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=7,
        speckleWindowSize=120,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp_l = stereo_l.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disp_r = stereo_r.compute(gray_r, gray_l).astype(np.float32) / 16.0

    valid = disp_l > min_disp

    yy, xx = np.indices(disp_l.shape)
    xr = np.round(xx - disp_l).astype(np.int32)

    consistency = np.zeros_like(valid)
    inside = valid & (xr >= 0) & (xr < w)

    diff = np.zeros_like(disp_l)
    diff[inside] = np.abs(disp_l[inside] + disp_r[yy[inside], xr[inside]])
    consistency[inside] = diff[inside] < 2.0

    valid = valid & consistency

    disp_norm = np.zeros_like(disp_l, dtype=np.uint8)

    if np.count_nonzero(valid) > 0.01 * disp_l.size:
        values = disp_l[valid]
        lo, hi = np.percentile(values, [3, 97])
        disp_norm = np.clip((disp_l - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255)
        disp_norm = disp_norm.astype(np.uint8)

    return disp_l, disp_norm, valid


def largest_components(mask, min_area_ratio=0.008, keep=3):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]

    min_area = min_area_ratio * mask.shape[0] * mask.shape[1]
    clean = np.zeros_like(mask)

    kept = 0
    for idx in order:
        label = idx + 1
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        clean[labels == label] = 255
        kept += 1

        if kept >= keep:
            break

    return clean


def disparity_mask(disparity, disp_norm, valid):
    mask = np.zeros(disparity.shape, dtype=np.uint8)

    if np.count_nonzero(valid) < 0.01 * disparity.size:
        return mask

    values = disparity[valid]

    threshold = np.percentile(values, 60)
    mask[(disparity >= threshold) & valid] = 255

    mask = cv2.medianBlur(mask, 7)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    mask = largest_components(mask, min_area_ratio=0.01, keep=2)

    return mask


def fallback_grabcut_mask(img):
    h, w = img.shape[:2]

    rect = (
        int(w * 0.10),
        int(h * 0.06),
        int(w * 0.80),
        int(h * 0.88)
    )

    gc_mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img,
        gc_mask,
        rect,
        bg_model,
        fg_model,
        8,
        cv2.GC_INIT_WITH_RECT
    )

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return mask


def refine_mask_with_grabcut(img, initial_mask):
    h, w = initial_mask.shape

    if np.count_nonzero(initial_mask) < 0.005 * initial_mask.size:
        return fallback_grabcut_mask(img)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    fg = initial_mask > 0
    bg = initial_mask == 0

    gc_mask[bg] = cv2.GC_PR_BGD
    gc_mask[fg] = cv2.GC_PR_FGD

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    sure_fg = cv2.erode(initial_mask, erode_kernel, iterations=1) > 0

    border = np.zeros_like(initial_mask, dtype=np.uint8)
    border[:8, :] = 255
    border[-8:, :] = 255
    border[:, :8] = 255
    border[:, -8:] = 255

    sure_bg = border > 0

    gc_mask[sure_fg] = cv2.GC_FGD
    gc_mask[sure_bg] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img,
        gc_mask,
        None,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    refined = largest_components(refined, min_area_ratio=0.01, keep=2)

    return refined


def make_soft_alpha(mask):
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel, iterations=1)

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (61, 61), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha[..., None]


def apply_portrait_effect(img, mask):
    blur_strong = cv2.GaussianBlur(img, (0, 0), sigmaX=24, sigmaY=24)
    blur_soft = cv2.GaussianBlur(img, (0, 0), sigmaX=10, sigmaY=10)

    background = cv2.addWeighted(blur_strong, 0.75, blur_soft, 0.25, 0)

    alpha = make_soft_alpha(mask)

    result = (
        alpha * img.astype(np.float32) +
        (1.0 - alpha) * background.astype(np.float32)
    )

    return np.clip(result, 0, 255).astype(np.uint8), background


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_and_save_grid(name, left, right, disp_color, mask, blurred_bg, result):
    OUT_DIR.mkdir(exist_ok=True)

    cv2.imwrite(str(OUT_DIR / f"{name}_01_esquerda_retificada.png"), left)
    cv2.imwrite(str(OUT_DIR / f"{name}_02_direita_retificada.png"), right)
    cv2.imwrite(str(OUT_DIR / f"{name}_03_disparidade.png"), disp_color)
    cv2.imwrite(str(OUT_DIR / f"{name}_04_mascara.png"), mask)
    cv2.imwrite(str(OUT_DIR / f"{name}_05_fundo_borrado.png"), blurred_bg)
    cv2.imwrite(str(OUT_DIR / f"{name}_06_modo_retrato.png"), result)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle(
        f"Questao 4 - Modo Retrato por Disparidade: {name}",
        fontsize=15,
        fontweight="bold"
    )

    items = [
        ("Imagem base retificada", bgr_to_rgb(left), None),
        ("Imagem auxiliar retificada", bgr_to_rgb(right), None),
        ("Mapa de disparidade", bgr_to_rgb(disp_color), None),
        ("Mascara refinada", mask, "gray"),
        ("Background borrado", bgr_to_rgb(blurred_bg), None),
        ("Resultado final", bgr_to_rgb(result), None),
    ]

    for ax, (title, img, cmap) in zip(axes.flat, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    grid_path = OUT_DIR / f"{name}_grade_completa.png"
    plt.savefig(str(grid_path), dpi=160, bbox_inches="tight")

    print(f"  grade salva em: {grid_path}")


def process_pair(name, left_file, right_file):
    print(f"\nProcessando par: {name}")

    img_l = load_image(left_file)
    img_r = load_image(right_file)

    if img_l.shape[:2] != img_r.shape[:2]:
        img_r = cv2.resize(
            img_r,
            (img_l.shape[1], img_l.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    try:
        left_rect, right_rect = rectify_uncalibrated(img_l, img_r)
    except Exception as exc:
        print(f"  retificacao falhou ({exc}); usando imagens originais.")
        left_rect, right_rect = img_l, img_r

    disparity, disp_norm, valid = compute_disparity(left_rect, right_rect)

    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    initial_mask = disparity_mask(disparity, disp_norm, valid)
    refined_mask = refine_mask_with_grabcut(left_rect, initial_mask)

    result, blurred_bg = apply_portrait_effect(left_rect, refined_mask)

    show_and_save_grid(
        name,
        left_rect,
        right_rect,
        disp_color,
        refined_mask,
        blurred_bg,
        result
    )


def main():
    print("Questao 4 - modo retrato usando mapa de disparidade")
    print("Pares usados:")
    for name, left, right in PAIRS:
        print(f"  {name}: {left} + {right}")

    for name, left_file, right_file in PAIRS:
        try:
            process_pair(name, left_file, right_file)
        except Exception as exc:
            print(f"  ERRO no par {name}: {exc}")

    print("\nFinalizado.")
    print(f"Resultados salvos em: {OUT_DIR}")

    plt.show()


if __name__ == "__main__":
    main()