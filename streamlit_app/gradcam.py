"""Mapa de atenção (Grad-CAM++ suavizado): mostra quais regiões da radiografia
mais influenciaram a probabilidade de fratura estimada pela CNN.

- Grad-CAM++ pondera cada filtro da última camada convolucional pelos
  gradientes positivos de P(fratura), o que costuma localizar melhor regiões
  pequenas do que o Grad-CAM simples.
- Suavização (SmoothGrad): o mapa é a média de várias cópias da imagem com um
  pouco de ruído, o que reduz manchas aleatórias.

Isto é uma ferramenta de **interpretabilidade visual**, não uma segmentação
médica da fratura: o mapa vem da última camada convolucional (grade 52×52,
ampliada para o tamanho da imagem), então a localização é aproximada. O mapa é
sempre calculado para a classe Fratura ("onde o modelo viu sinais de
fratura"), inclusive quando o resultado final é Sem fratura.
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

SMOOTH_SAMPLES = 12      # cópias com ruído além da imagem original
NOISE_STD = 0.08         # desvio do ruído, em fração da faixa 0-255
SEED = 42
NORMALIZE_PERCENTILE = 99
VISIBLE_FROM = 0.15      # abaixo disso o mapa fica transparente
HOTSPOT_LEVEL = 0.8      # regiões >= 80% da escala recebem caixa
MAX_BOXES = 3
MIN_BOX_CELLS = 3        # ignora regiões muito pequenas (células da grade 52×52)


def _last_conv_index(model: tf.keras.Model) -> int:
    for i in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            return i
    raise ValueError("Nenhuma camada Conv2D encontrada no modelo.")


def compute_gradcam_heatmap(model: tf.keras.Model, preprocessed_batch) -> np.ndarray:
    """Recebe um lote (1, H, W, 3) com pixels originais (0-255) — o modelo já
    normaliza internamente — e devolve um heatmap 2D normalizado em [0, 1]."""
    image = tf.convert_to_tensor(preprocessed_batch, tf.float32)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 255 * NOISE_STD, (SMOOTH_SAMPLES,) + tuple(image.shape[1:])).astype("float32")
    batch = tf.concat([image, tf.clip_by_value(image + noise, 0, 255)], axis=0)

    last_conv = _last_conv_index(model)
    with tf.GradientTape() as tape:
        x = batch
        conv_output = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)  # augmentation e Dropout desligados
            if i == last_conv:
                conv_output = x
                tape.watch(conv_output)
        score = tf.reduce_sum(x[:, 0])  # P(fratura) de cada cópia (gradientes independentes)

    grads = tape.gradient(score, conv_output).numpy()
    acts = conv_output.numpy()

    # Pesos do Grad-CAM++ (Chattopadhay et al., 2018), por cópia e por filtro.
    g2, g3 = grads ** 2, grads ** 3
    denom = 2 * g2 + acts.sum(axis=(1, 2), keepdims=True) * g3
    alpha = g2 / np.where(denom != 0, denom, 1e-8)
    weights = (alpha * np.maximum(grads, 0)).sum(axis=(1, 2))           # (N, filtros)
    cams = np.maximum(np.einsum("nhwc,nc->nhw", acts, weights), 0)       # (N, H, W)

    # Normaliza pelo percentil 99 (e não pelo máximo): assim um pico isolado,
    # como uma letra de lateralidade, não apaga o restante do mapa.
    heatmap = cams.mean(axis=0)
    reference = np.percentile(heatmap, NORMALIZE_PERCENTILE)
    if reference <= 0:
        reference = heatmap.max()
    return np.clip(heatmap / reference, 0, 1) if reference > 0 else heatmap


def _turbo(values: np.ndarray) -> np.ndarray:
    """Mapa de cores "turbo" (azul -> verde -> amarelo -> vermelho), pela
    aproximação polinomial publicada pelo Google; evita depender do matplotlib."""
    x = np.clip(values, 0, 1)[..., None]
    r = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
    g = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    b = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])
    powers = np.concatenate([x ** k for k in range(6)], axis=-1)
    rgb = np.stack([powers @ r, powers @ g, powers @ b], axis=-1)
    return np.clip(rgb, 0, 1) * 255


def hotspot_boxes(heatmap: np.ndarray, level: float = HOTSPOT_LEVEL, max_boxes: int = MAX_BOXES):
    """Caixas (na grade do heatmap) das regiões conectadas com ativação >=
    level, da mais forte para a mais fraca. Devolve [(x0, y0, x1, y1), ...]."""
    mask = heatmap >= level
    seen = np.zeros_like(mask)
    regions = []
    h, w = mask.shape
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack, cells = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(cells) >= MIN_BOX_CELLS:
            ys, xs = zip(*cells)
            strength = float(sum(heatmap[c] for c in cells))
            regions.append((strength, (min(xs), min(ys), max(xs), max(ys))))
    regions.sort(reverse=True)
    return [box for _, box in regions[:max_boxes]]


def overlay_heatmap_on_image(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    opacity: float = 0.55,
    draw_boxes: bool = True,
) -> Image.Image:
    """Sobrepõe o heatmap (cores turbo, transparente onde a influência é baixa)
    na imagem original e desenha caixas nas regiões de maior influência."""
    img = pil_image.convert("RGB")
    width, height = img.size

    heat = Image.fromarray(np.uint8(255 * heatmap)).resize((width, height), Image.BICUBIC)
    heat = np.asarray(heat, dtype=np.float32) / 255.0
    visible = np.clip((heat - VISIBLE_FROM) / (1 - VISIBLE_FROM), 0, 1)
    alpha = (visible * opacity)[..., None]

    base = np.asarray(img, dtype=np.float32)
    blended = base * (1 - alpha) + _turbo(heat) * alpha
    result = Image.fromarray(np.uint8(blended))

    if draw_boxes:
        draw = ImageDraw.Draw(result)
        grid_h, grid_w = heatmap.shape
        sx, sy = width / grid_w, height / grid_h
        line = max(2, width // 150)
        font = ImageFont.load_default(size=max(12, width // 28))
        for n, (x0, y0, x1, y1) in enumerate(hotspot_boxes(heatmap), start=1):
            rect = (x0 * sx, y0 * sy, (x1 + 1) * sx, (y1 + 1) * sy)
            draw.rectangle(rect, outline=(0, 0, 0), width=line + 2)      # contorno escuro p/ contraste
            draw.rectangle(rect, outline=(255, 255, 255), width=line)
            tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), str(n), font=font)
            pad = line + 2
            draw.rectangle((rect[0], rect[1], rect[0] + tx1 - tx0 + 2 * pad, rect[1] + ty1 - ty0 + 2 * pad),
                           fill=(255, 255, 255))
            draw.text((rect[0] + pad - tx0, rect[1] + pad - ty0), str(n), fill=(0, 0, 0), font=font)
    return result
