"""Grad-CAM (Gradient-weighted Class Activation Mapping): gera um mapa de
calor que mostra quais regiões da radiografia mais influenciaram a predição
do modelo, e uma caixa delimitadora ao redor da região de maior ativação.

Isto é uma ferramenta de **interpretabilidade visual**, não uma segmentação
médica da fratura: o mapa é calculado a partir da última camada convolucional
da CNN (grade 52×52, depois ampliada para o tamanho da imagem), então a
localização é aproximada. Serve para o radiologista entender "para onde o
modelo olhou", não para apontar com precisão anatômica a linha de fratura.
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


def _last_conv_index(model: tf.keras.Model) -> int:
    for i in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            return i
    raise ValueError("Nenhuma camada Conv2D encontrada no modelo.")


def compute_gradcam_heatmap(model: tf.keras.Model, preprocessed_batch) -> np.ndarray:
    """Recebe um lote (1, H, W, 3) com pixels originais (0-255) — o próprio
    modelo salvo já normaliza internamente — e devolve um heatmap 2D
    (resolução da última camada convolucional) normalizado em [0, 1], onde
    valores maiores indicam maior influência na predição de 'Fraturado'.

    As camadas são chamadas uma a uma em modo eager (training=False, então o
    data augmentation e o Dropout ficam desligados), guardando a saída da
    última Conv2D para calcular o gradiente em relação a ela.
    """
    last_conv = _last_conv_index(model)
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(preprocessed_batch)
        conv_output = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if i == last_conv:
                conv_output = x
                tape.watch(conv_output)
        loss = x[:, 0]  # saída sigmoid única = P(fratura)

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.tensordot(conv_output, pooled_grads, axes=[[2], [0]])
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy()


def _hex_to_rgb(color_hex: str):
    color_hex = color_hex.lstrip("#")
    return tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))


def bounding_box_from_heatmap(heatmap: np.ndarray, threshold: float = 0.6):
    """Caixa delimitadora (em coordenadas da grade do heatmap) ao redor dos
    pontos com ativação >= threshold do máximo. Retorna None se nenhum ponto
    atingir o limiar."""
    mask = heatmap >= threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay_heatmap_on_image(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    color_hex: str,
    alpha_max: float = 0.55,
    draw_box: bool = True,
) -> Image.Image:
    """Sobrepõe o heatmap (um único tom, intensidade = ativação) na imagem
    original e desenha uma caixa ao redor da região de maior ativação."""
    img = pil_image.convert("RGB")
    width, height = img.size

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((width, height), Image.BICUBIC)
    heatmap_arr = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    color_rgb = np.array(_hex_to_rgb(color_hex), dtype=np.float32)
    base_arr = np.array(img, dtype=np.float32)
    alpha = (heatmap_arr * alpha_max)[..., np.newaxis]
    color_layer = np.ones_like(base_arr) * color_rgb
    blended = base_arr * (1 - alpha) + color_layer * alpha
    result = Image.fromarray(np.uint8(blended))

    if draw_box:
        box = bounding_box_from_heatmap(heatmap, threshold=0.6)
        if box is not None:
            grid_h, grid_w = heatmap.shape
            x0, y0, x1, y1 = box
            scale_x, scale_y = width / grid_w, height / grid_h
            rect = (
                x0 * scale_x,
                y0 * scale_y,
                (x1 + 1) * scale_x,
                (y1 + 1) * scale_y,
            )
            draw = ImageDraw.Draw(result)
            draw.rectangle(rect, outline=_hex_to_rgb(color_hex), width=max(3, width // 100))

    return result
