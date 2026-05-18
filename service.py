import base64
import io
import os
import random
import warnings
from typing import Optional

import PIL.Image
import legacy
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

G = None
SEGMENTATION_MODEL = None
PETROGLYPH_IMPORT_ERROR = None
petroglyph = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DEFAULT_SEGMENTATION_THRESHOLD = 0.5
DEFAULT_SEGMENTATION_MIN_AREA = 150
DEFAULT_SEGMENTATION_LINE_WIDTH = 2

try:
    import segmentar_petroglifo as petroglyph  # type: ignore[import-not-found]
except Exception as exc:
    PETROGLYPH_IMPORT_ERROR = exc
    petroglyph = None
else:
    DEFAULT_SEGMENTATION_THRESHOLD = petroglyph.DEFAULT_THRESHOLD
    DEFAULT_SEGMENTATION_MIN_AREA = petroglyph.DEFAULT_MIN_AREA
    DEFAULT_SEGMENTATION_LINE_WIDTH = petroglyph.DEFAULT_LINE_WIDTH


class ImageRequest(BaseModel):
    seed: Optional[float] = 456
    truncation_psi: float = 1.0
    noise_mode: str = "const"
    number: Optional[int] = 1


def load_gan_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    with open(model_path, "rb") as f:
        return legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore[index]


def pil_to_b64(pil_img: PIL.Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def array_to_b64(image_array: np.ndarray) -> str:
    pil_img = PIL.Image.fromarray(image_array)
    return pil_to_b64(pil_img)


def grayscale_to_b64(image_array: np.ndarray) -> str:
    pil_img = PIL.Image.fromarray(image_array.astype(np.uint8), mode="L")
    return pil_to_b64(pil_img)


def build_segmentation_status(metrics: dict, strategy: str) -> str:
    warnings = set(metrics.get("warnings", []))
    if "mask_empty" in warnings or "mask_too_small" in warnings:
        return "weak_segmentation"
    if "mask_too_large" in warnings:
        return "oversegmented"
    if strategy == "refined":
        return "refined_mask"
    return "ok"


@app.on_event("startup")
async def startup_event():
    global G, SEGMENTATION_MODEL

    gan_model_path = "modelo/pictos512.pkl"

    print(f"Cargando modelo GAN desde {gan_model_path} ...")
    try:
        G = load_gan_model(gan_model_path)
        print("Modelo GAN cargado correctamente.")
    except Exception as e:
        print(f"Error cargando el modelo GAN: {e}")

    if petroglyph is None:
        print(f"Segmentacion de petroglifos deshabilitada: {PETROGLYPH_IMPORT_ERROR}")
        return

    segmentation_model_path = str(petroglyph.MODEL_PATH)
    print(f"Cargando modelo de segmentacion desde {segmentation_model_path} ...")
    try:
        SEGMENTATION_MODEL = petroglyph.load_model(
            segmentation_model_path,
            custom_objects={
                "bce_dice_loss": petroglyph.bce_dice_loss,
                "dice_coefficient": petroglyph.dice_coefficient,
                "iou_score": petroglyph.iou_score,
            },
        )
        print("Modelo de segmentacion cargado correctamente.")
    except Exception as e:
        print(f"Error cargando el modelo de segmentacion: {e}")


@app.post("/generateSingle")
async def generate_image(image: ImageRequest):
    seed = int(image.seed)
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode

    global G
    if G is None:
        raise HTTPException(status_code=500, detail="Modelo GAN no cargado")

    label = torch.zeros([1, G.c_dim], device=device)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

    return {
        "image": pil_to_b64(pil_img),
        "seed": seed,
        "truncation_psi": truncation_psi,
        "noise_mode": noise_mode,
    }


@app.post("/generateSeveral")
async def generate_several(image: ImageRequest):
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode

    global G
    if G is None:
        raise HTTPException(status_code=500, detail="Modelo GAN no cargado")

    label = torch.zeros([1, G.c_dim], device=device)
    images = []
    seeds = []

    for _ in range(0, image.number, 1):
        seed = random.randint(1, 2147483647)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

        images.append(pil_to_b64(pil_img))
        seeds.append(seed)

    return {
        "number": image.number,
        "images": images,
        "seeds": seeds,
        "truncation_psi": truncation_psi,
        "noise_mode": noise_mode,
    }


@app.post("/segmentPetroglyph")
async def segment_petroglyph(
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_SEGMENTATION_THRESHOLD),
    min_area: int = Form(DEFAULT_SEGMENTATION_MIN_AREA),
    line_width: int = Form(DEFAULT_SEGMENTATION_LINE_WIDTH),
    include_previews: bool = Form(True),
):
    global SEGMENTATION_MODEL

    if petroglyph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "La segmentacion no esta disponible porque faltan dependencias del modelo "
                f"({PETROGLYPH_IMPORT_ERROR})."
            ),
        )

    if SEGMENTATION_MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo de segmentacion no cargado")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Debes enviar un archivo de imagen valido")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="La imagen enviada esta vacia")

    np_buffer = np.frombuffer(content, np.uint8)
    img_bgr = petroglyph.cv2.imdecode(np_buffer, petroglyph.cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada")

    img_pre = petroglyph.preprocess(img_bgr)
    probability = petroglyph.predict_tta(SEGMENTATION_MODEL, img_pre)

    selected_mask = petroglyph.select_best_mask(probability, threshold, min_area)
    filled_mask = selected_mask["mask"]
    line_mask = petroglyph.adelgazar_mascara(filled_mask, grosor=line_width)
    metrics = selected_mask["metrics"]

    original_rgb = petroglyph.cv2.cvtColor(
        petroglyph.cv2.resize(img_bgr, (petroglyph.IMG_SIZE, petroglyph.IMG_SIZE), interpolation=petroglyph.cv2.INTER_AREA),
        petroglyph.cv2.COLOR_BGR2RGB,
    )
    background_rgb, background_name = petroglyph.choose_background(None)
    rendered_rgb = petroglyph.render_petroglyph(background_rgb, filled_mask)

    area_percent = float(metrics["area_percent"])
    segmentation_status = build_segmentation_status(metrics, str(selected_mask["strategy"]))

    response = {
        "filename": file.filename,
        "background_name": background_name,
        "threshold": threshold,
        "min_area": min_area,
        "line_width": line_width,
        "area_percent": round(area_percent, 2),
        "segmentation_status": segmentation_status,
        "selected_threshold": float(selected_mask["threshold"]),
        "selected_strategy": str(selected_mask["strategy"]),
        "validation_score": round(float(metrics["score"]), 2),
        "validation_warnings": list(metrics["warnings"]),
        "component_count": int(metrics["component_count"]),
        "size": {
            "width": petroglyph.IMG_SIZE,
            "height": petroglyph.IMG_SIZE,
        },
        "result_image": array_to_b64(rendered_rgb),
        "background_image": array_to_b64(background_rgb),
        "mask_image": grayscale_to_b64(filled_mask),
    }

    if include_previews:
        probability_img = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
        response.update(
            {
                "original_image": array_to_b64(original_rgb),
                "probability_image": grayscale_to_b64(probability_img),
                "line_image": grayscale_to_b64(line_mask),
            }
        )

    return response
