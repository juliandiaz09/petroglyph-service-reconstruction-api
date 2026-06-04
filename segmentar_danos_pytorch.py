from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "modelo" / "unet_danos_petroglifos_v3.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMG_SIZE = 512
DEFAULT_THRESHOLD = 0.30
DEFAULT_MIN_AREA = 50
DEFAULT_OPEN_KERNEL = 3
DEFAULT_CLOSE_KERNEL = 7
DEFAULT_AGGRESSIVE_TARGET_PERCENT = 10.0
DEFAULT_AGGRESSIVE_MIN_PIXELS = 2500
DEFAULT_AGGRESSIVE_DILATION = 2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(model_path: str | Path = MODEL_PATH, device: str | None = None) -> tuple[torch.nn.Module, torch.device]:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    resolved_device = resolve_device(device)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    state_dict = torch.load(model_path, map_location=resolved_device)
    model.load_state_dict(state_dict, strict=True)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr debe ser una imagen BGR de 3 canales")

    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).unsqueeze(0)
    return tensor


def fill_holes(mask_bin: np.ndarray) -> np.ndarray:
    if mask_bin.ndim != 2:
        raise ValueError("mask_bin debe ser una mascara de 2 dimensiones")

    h, w = mask_bin.shape
    flood = mask_bin.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_bin, flood_inv)


def postprocess_mask(mask_bin: np.ndarray, min_area: int = DEFAULT_MIN_AREA) -> np.ndarray:
    if mask_bin.ndim != 2:
        raise ValueError("mask_bin debe ser una mascara de 2 dimensiones")

    mask = (mask_bin > 0).astype(np.uint8) * 255
    if not np.any(mask):
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    clean = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == idx] = 255

    if np.any(clean):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DEFAULT_CLOSE_KERNEL, DEFAULT_CLOSE_KERNEL))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        clean = cv2.dilate(clean, dilate_kernel, iterations=2)

    return clean


def build_aggressive_mask(
    probability: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    target_percent: float = DEFAULT_AGGRESSIVE_TARGET_PERCENT,
    min_pixels: int = DEFAULT_AGGRESSIVE_MIN_PIXELS,
    dilation_iters: int = DEFAULT_AGGRESSIVE_DILATION,
) -> np.ndarray:
    """
    Construye una mascara mas agresiva cuando la segmentacion normal sale vacia.

    Primero intenta el flujo normal. Si la mascara queda vacia o demasiado pequena,
    toma los pixeles con mayor probabilidad hasta cubrir un porcentaje objetivo del
    cuadro y luego los expande levemente para asegurar continuidad.
    """
    binary = (probability >= threshold).astype(np.uint8) * 255
    clean = postprocess_mask(binary, min_area=min_area)
    if np.any(clean):
        return clean

    flat = probability.reshape(-1)
    total_pixels = int(flat.size)
    target_pixels = int(max(min_pixels, total_pixels * max(target_percent, 0.01) / 100.0))
    target_pixels = max(1, min(total_pixels, target_pixels))

    if target_pixels >= total_pixels:
        mask = np.ones_like(probability, dtype=np.uint8) * 255
    else:
        cutoff_index = total_pixels - target_pixels
        cutoff_value = np.partition(flat, cutoff_index)[cutoff_index]
        mask = (probability >= cutoff_value).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (DEFAULT_CLOSE_KERNEL, DEFAULT_CLOSE_KERNEL),
    )
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, dilate_kernel, iterations=max(0, int(dilation_iters)))

    return mask


def predict_probability(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = preprocess_image(img_bgr).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    return probability.astype(np.float32)


def predict_mask(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    aggressive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    probability = predict_probability(model, img_bgr, device)
    binary = (probability >= threshold).astype(np.uint8) * 255
    clean = postprocess_mask(binary, min_area=min_area)
    if aggressive and not np.any(clean):
        clean = build_aggressive_mask(probability, threshold=threshold, min_area=min_area)
    return probability, clean


def save_mask_png(mask: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), mask.astype(np.uint8))
    if not ok:
        raise IOError(f"No se pudo guardar la mascara en {output_path}")
    return output_path


def segment_image_file(
    image_path: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
    model_path: str | Path = MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    aggressive: bool = False,
    device: str | None = None,
) -> dict[str, object]:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    model, resolved_device = load_model(model_path=model_path, device=device)
    probability, mask = predict_mask(
        model=model,
        img_bgr=img_bgr,
        device=resolved_device,
        threshold=threshold,
        min_area=min_area,
        aggressive=aggressive,
    )

    output_dir = Path(output_dir)
    mask_path = output_dir / f"{image_path.stem}_mascara.png"
    save_mask_png(mask, mask_path)

    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "threshold": float(threshold),
        "min_area": int(min_area),
        "device": str(resolved_device),
        "probability_shape": list(probability.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta danos en petroglifos con U-Net ResNet34 en PyTorch."
    )
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada.")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Ruta al archivo .pth con los pesos del modelo.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Carpeta donde se guardara la mascara PNG.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Umbral de binarizacion.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Area minima por componente conectado.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo de inferencia: auto, cpu o cuda.",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Usa un fallback agresivo si la mascara normal sale vacia.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = segment_image_file(
        image_path=args.image,
        output_dir=args.output_dir,
        model_path=args.model,
        threshold=args.threshold,
        min_area=args.min_area,
        aggressive=args.aggressive,
        device=args.device,
    )

    print(f"Mask guardada en: {result['mask_path']}")
    print(f"Device: {result['device']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Min area: {result['min_area']}")


if __name__ == "__main__":
    main()
