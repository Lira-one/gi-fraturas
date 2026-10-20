"""Grad-CAM (Gradient-weighted Class Activation Mapping): gera um mapa de
calor que mostra quais regiões da radiografia mais influenciaram a predição
do modelo, e uma caixa delimitadora ao redor da região de maior ativação.

Isto é uma ferramenta de **interpretabilidade visual**, não uma segmentação
médica da fratura: o mapa é calculado a partir da última camada convolucional
do MobileNetV2 (grade 7×7, depois ampliada para o tamanho da imagem), então a
localização é aproximada e de baixa resolução espacial. Serve para o
radiologista entender "para onde o modelo olhou", não para apontar com
precisão anatômica a linha de fratura.
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


def _find_base_conv_model(model) -> tf.keras.Model:
    """Encontra o modelo convolucional aninhado (MobileNetV2) dentre as
    camadas do modelo treinado — é o que tem mais camadas (a cabeça de
    data_augmentation também é um modelo aninhado, mas tem só 3 camadas)."""
    nested_models = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not nested_models:
        raise ValueError("Nenhum modelo convolucional aninhado encontrado.")
    return max(nested_models, key=lambda l: len(l.layers))


def _find_last_conv_layer_name(base_model: tf.keras.Model) -> str:
    for layer in base_model.layers:
        if layer.name == "out_relu":
            return layer.name
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except (AttributeError, ValueError):
            continue
    raise ValueError("Nenhuma camada convolucional 4D encontrada na base.")


def _split_layers_around_base(model: tf.keras.Model, base_model: tf.keras.Model):
    """Separa as camadas de nível superior do modelo treinado em 'antes' e
    'depois' da base convolucional (ex.: antes = data_augmentation +
    Rescaling; depois = GlobalAveragePooling2D + Dropout + Dense). A
    InputLayer é descartada (é só um marcador simbólico, não se chama).
    """
    pre_layers, post_layers = [], []
    seen_base = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer is base_model:
            seen_base = True
            continue
        (post_layers if seen_base else pre_layers).append(layer)
    return pre_layers, post_layers


def compute_gradcam_heatmap(model: tf.keras.Model, preprocessed_batch) -> np.ndarray:
    """Recebe um lote (1, H, W, 3) com pixels originais (0-255) — o próprio
    modelo salvo já normaliza internamente — e devolve um heatmap 2D
    (resolução da última camada convolucional, ex. 7x7) normalizado em
    [0, 1], onde valores maiores indicam maior influência na predição de
    'Fraturado'.

    Implementado chamando as camadas uma a uma em modo eager (em vez de
    montar um novo `tf.keras.Model(inputs=..., outputs=[...])` a partir de
    camadas de um modelo aninhado): no Keras 3, construir um Model assim a
    partir da saída de uma camada interna de um submodelo já chamado levanta
    `ValueError: Output ... is not connected to inputs`, mesmo quando a
    conexão existe de fato no grafo de execução eager.
    """
    base_model = _find_base_conv_model(model)
    last_conv_layer_name = _find_last_conv_layer_name(base_model)
    conv_submodel = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(last_conv_layer_name).output,
    )
    pre_layers, post_layers = _split_layers_around_base(model, base_model)

    x = preprocessed_batch
    for layer in pre_layers:
        x = layer(x, training=False)

    with tf.GradientTape() as tape:
        tape.watch(x)
        conv_output = conv_submodel(x, training=False)
        y = conv_output
        for layer in post_layers:
            y = layer(y, training=False)
        loss = y[:, 0]  # saída sigmoid única = P(fratura)

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
