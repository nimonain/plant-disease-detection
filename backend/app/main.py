from io import BytesIO
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.model import (
    ModelNotReadyError,
    ModelMetadata,
    PredictionModel,
    load_cnn_model,
    load_knn_model,
    load_mobilenetv2_model,
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 5
SUPPORTED_MODEL_IDS = ("knn", "cnn", "mobilenet")
MODEL_ID_PATTERN = f"^({'|'.join(SUPPORTED_MODEL_IDS)})$"
DISEASE_KNOWLEDGE_BASE_PATH = (
    Path(__file__).parent / "prevention" / "plantvillage_disease_knowledge_base.json"
)
DEVELOPMENT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class HealthResponse(BaseModel):
    status: str
    service_ready: bool
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_id: str
    architecture: str
    version: str
    supported_classes: list[str]
    ready: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfoResponse]


class DiseaseCauseResponse(BaseModel):
    type: str
    organism: str | None


class DiseaseInformationResponse(BaseModel):
    class_name: str
    crop: str
    disease_name: str
    status: str
    cause: DiseaseCauseResponse
    description: str
    symptoms: list[str]
    prevention: list[str]
    recommended_actions: list[str]
    severity: str
    spread_risk: str
    treatment_note: str
    disclaimer: str


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    model_version: str
    disease_information: DiseaseInformationResponse | None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


def load_disease_knowledge_base(
    path: Path = DISEASE_KNOWLEDGE_BASE_PATH,
) -> dict[str, DiseaseInformationResponse]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("diseases"), list):
        raise ValueError(f"Disease knowledge base has an invalid structure: {path}")

    knowledge_by_class: dict[str, DiseaseInformationResponse] = {}
    for entry in payload["diseases"]:
        if not isinstance(entry, dict):
            raise ValueError(f"Disease knowledge entries must be objects: {path}")
        information = DiseaseInformationResponse(
            class_name=entry["className"],
            crop=entry["crop"],
            disease_name=entry["diseaseName"],
            status=entry["status"],
            cause=DiseaseCauseResponse(**entry["cause"]),
            description=entry["description"],
            symptoms=entry["symptoms"],
            prevention=entry["prevention"],
            recommended_actions=entry["recommendedActions"],
            severity=entry["severity"],
            spread_risk=entry["spreadRisk"],
            treatment_note=entry["treatmentNote"],
            disclaimer=entry["disclaimer"],
        )
        if information.class_name in knowledge_by_class:
            raise ValueError(
                f"Duplicate disease class in knowledge base: {information.class_name}"
            )
        knowledge_by_class[information.class_name] = information

    if not knowledge_by_class:
        raise ValueError(f"Disease knowledge base contains no disease entries: {path}")
    return knowledge_by_class


def get_cors_origins() -> list[str]:
    app_environment = os.getenv("APP_ENV", "development").strip().lower()
    configured_origins = os.getenv("CORS_ORIGINS")
    if configured_origins is None:
        if app_environment == "production":
            raise ValueError("CORS_ORIGINS must be set in production.")
        return list(DEVELOPMENT_CORS_ORIGINS)

    origins = [origin.strip().rstrip("/") for origin in configured_origins.split(",")]
    origins = [origin for origin in origins if origin]
    if not origins:
        raise ValueError("CORS_ORIGINS must contain at least one allowed origin.")
    if "*" in origins:
        raise ValueError("CORS_ORIGINS must list explicit origins, not '*'.")
    if app_environment == "production":
        for origin in origins:
            parsed_origin = urlsplit(origin)
            if (
                parsed_origin.scheme != "https"
                or not parsed_origin.netloc
                or parsed_origin.path
                or parsed_origin.query
                or parsed_origin.fragment
                or parsed_origin.username
                or parsed_origin.password
            ):
                raise ValueError(
                    "Production CORS_ORIGINS must contain HTTPS origins only."
                )
    return origins


def create_app(
    model: PredictionModel | None = None,
    *,
    models: dict[str, PredictionModel] | None = None,
) -> FastAPI:
    if model is not None and models is not None:
        raise ValueError("Pass either a single model or a model registry, not both.")
    model_registry = models if models is not None else (
        {"knn": model}
        if model is not None
        else {
            "knn": load_knn_model(),
            "cnn": load_cnn_model(),
            "mobilenet": load_mobilenetv2_model(),
        }
    )
    if not model_registry:
        raise ValueError("At least one prediction model must be configured.")
    if any(model_id not in SUPPORTED_MODEL_IDS for model_id in model_registry):
        raise ValueError(f"Model IDs must be one of {SUPPORTED_MODEL_IDS}.")
    disease_knowledge = load_disease_knowledge_base()

    api = FastAPI(
        title="Leaf Disease Prediction API",
        version="1.0.0",
        description="HTTP API for classifying leaf images.",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def get_model(model_id: str) -> PredictionModel:
        try:
            return model_registry[model_id]
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' is not available.",
            ) from error

    def metadata_response(model_id: str, metadata: ModelMetadata) -> ModelInfoResponse:
        return ModelInfoResponse(model_id=model_id, **metadata.__dict__)

    def make_prediction(
        image: Image.Image, model_id: str
    ) -> PredictionResponse:
        prediction_model = get_model(model_id)
        try:
            result = prediction_model.predict(image)
        except ModelNotReadyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return PredictionResponse(
            class_name=result.class_name,
            confidence=result.confidence,
            model_version=prediction_model.metadata.version,
            disease_information=disease_knowledge.get(result.class_name),
        )

    @api.get("/", tags=["root"])
    def root() -> dict[str, object]:
        return {
            "message": "Welcome to the Leaf Disease Prediction API.",
            "project": api.title,
            "version": api.version,
            "documentation": "/docs",
            "openapi_schema": "/openapi.json",
            "models": list(SUPPORTED_MODEL_IDS),
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/",
                    "description": "API welcome message and endpoint overview.",
                },
                {
                    "method": "GET",
                    "path": "/api/v1/health",
                    "description": "Check service and model readiness.",
                },
                {
                    "method": "GET",
                    "path": "/api/v1/models",
                    "description": "List configured models and their metadata.",
                },
                {
                    "method": "GET",
                    "path": "/api/v1/model/info",
                    "description": "Get metadata for a model.",
                    "query_parameters": {
                        "model_id": {
                            "options": list(SUPPORTED_MODEL_IDS),
                            "default": "knn",
                        }
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/v1/predict",
                    "description": "Classify one uploaded image.",
                    "form_fields": {
                        "file": "Image file (required; maximum 10 MiB).",
                        "model_id": {
                            "options": list(SUPPORTED_MODEL_IDS),
                            "default": "knn",
                        },
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/v1/predict/batch",
                    "description": "Classify up to 5 uploaded images.",
                    "form_fields": {
                        "files": "Repeated image files (required; maximum 10 MiB each).",
                        "model_id": {
                            "options": list(SUPPORTED_MODEL_IDS),
                            "default": "knn",
                        },
                    },
                },
            ],
        }

    @api.get("/health", response_model=HealthResponse, tags=["health"])
    @api.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    def health(response: Response) -> HealthResponse:
        model_ready = all(
            prediction_model.metadata.ready
            for prediction_model in model_registry.values()
        )
        if not model_ready:
            response.status_code = 503
        return HealthResponse(
            status="ready" if model_ready else "not_ready",
            service_ready=True,
            model_ready=model_ready,
        )

    @api.get(
        "/api/v1/model/info",
        response_model=ModelInfoResponse,
        tags=["model"],
    )
    def model_info(
        model_id: str = Query(default="knn", pattern=MODEL_ID_PATTERN),
    ) -> ModelInfoResponse:
        prediction_model = get_model(model_id)
        return metadata_response(model_id, prediction_model.metadata)

    @api.get("/api/v1/models", response_model=ModelsResponse, tags=["model"])
    def models_info() -> ModelsResponse:
        return ModelsResponse(
            models=[
                metadata_response(model_id, prediction_model.metadata)
                for model_id, prediction_model in model_registry.items()
            ]
        )

    @api.post(
        "/api/v1/predict",
        response_model=PredictionResponse,
        tags=["prediction"],
    )
    async def predict(
        file: UploadFile = File(...),
        model_id: str = Form(default="knn", pattern=MODEL_ID_PATTERN),
    ) -> PredictionResponse:
        image = await read_image(file)
        return make_prediction(image, model_id)

    @api.post(
        "/api/v1/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["prediction"],
    )
    async def predict_batch(
        files: list[UploadFile] = File(...),
        model_id: str = Form(default="knn", pattern=MODEL_ID_PATTERN),
    ) -> BatchPredictionResponse:
        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(
                status_code=413,
                detail=f"A batch can contain at most {MAX_BATCH_FILES} images.",
            )
        images = [await read_image(file) for file in files]
        return BatchPredictionResponse(
            predictions=[
                make_prediction(image, model_id) for image in images
            ]
        )

    return api


async def read_image(file: UploadFile) -> Image.Image:
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Each image must be {MAX_IMAGE_BYTES // (1024 * 1024)} MiB or smaller.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()
        with Image.open(BytesIO(contents)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=415,
            detail="The uploaded file is not a supported image.",
        ) from error


app = create_app()
