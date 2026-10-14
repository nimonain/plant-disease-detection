from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import (
    MAX_BATCH_FILES,
    MAX_IMAGE_BYTES,
    app,
    create_app,
    load_disease_knowledge_base,
)
from app.model import (
    CnnModel,
    KnnPcaModel,
    ModelMetadata,
    MobileNetV2Model,
    Prediction,
    UnconfiguredModel,
    load_cnn_model,
    load_knn_model,
    load_mobilenetv2_model,
)


@pytest.fixture(scope="module")
def knn_model() -> KnnPcaModel:
    return load_knn_model()


@pytest.fixture(scope="module")
def cnn_model() -> CnnModel:
    return load_cnn_model()


@pytest.fixture(scope="module")
def mobilenetv2_model() -> MobileNetV2Model:
    return load_mobilenetv2_model()


class StubModel:
    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            architecture="test-model",
            version="test-1",
            supported_classes=["healthy", "leaf-spot"],
            ready=True,
        )

    def predict(self, _: Image.Image) -> Prediction:
        return Prediction(class_name="healthy", confidence=0.98)


def png_image() -> bytes:
    image = Image.new("RGB", (2, 2), color="green")
    contents = BytesIO()
    image.save(contents, format="PNG")
    return contents.getvalue()


def test_health_reports_unconfigured_model_as_not_ready() -> None:
    response = TestClient(create_app(UnconfiguredModel())).get("/api/v1/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service_ready": True,
        "model_ready": False,
    }


def test_health_reports_configured_model_as_ready() -> None:
    response = TestClient(create_app(StubModel())).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service_ready": True,
        "model_ready": True,
    }


def test_health_alias_reports_configured_model_as_ready() -> None:
    response = TestClient(create_app(StubModel())).get("/health")

    assert response.status_code == 200
    assert response.json()["model_ready"] is True


def test_cors_allows_configured_production_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("CORS_ORIGINS", "https://leaf.example.com")
    client = TestClient(create_app(StubModel()))

    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://leaf.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://leaf.example.com"

    rejected_response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://untrusted.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert rejected_response.status_code == 400
    assert "access-control-allow-origin" not in rejected_response.headers


def test_production_requires_explicit_cors_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    with pytest.raises(ValueError, match="CORS_ORIGINS must be set in production"):
        create_app(StubModel())


def test_root_returns_welcome_and_api_overview() -> None:
    response = TestClient(create_app(StubModel())).get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Welcome to the Leaf Disease Prediction API."
    assert body["documentation"] == "/docs"
    assert body["models"] == ["knn", "cnn", "mobilenet"]
    assert {
        (endpoint["method"], endpoint["path"])
        for endpoint in body["endpoints"]
    } == {
        ("GET", "/"),
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/models"),
        ("GET", "/api/v1/model/info"),
        ("POST", "/api/v1/predict"),
        ("POST", "/api/v1/predict/batch"),
    }


def test_prediction_includes_matching_disease_information() -> None:
    class AppleScabModel(StubModel):
        def predict(self, _: Image.Image) -> Prediction:
            return Prediction(class_name="Apple___Apple_scab", confidence=0.91)

    response = TestClient(create_app(AppleScabModel())).post(
        "/api/v1/predict",
        files={"file": ("leaf.png", png_image(), "image/png")},
    )

    assert response.status_code == 200
    prediction = response.json()
    information = prediction["disease_information"]
    assert information["class_name"] == prediction["class_name"]
    assert information["disease_name"] == "Apple Scab"
    assert information["cause"]["organism"] == "Venturia inaequalis"
    assert information["prevention"]
    assert information["recommended_actions"]


def test_model_info_returns_configured_model_metadata() -> None:
    response = TestClient(create_app(StubModel())).get("/api/v1/model/info")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "knn",
        "architecture": "test-model",
        "version": "test-1",
        "supported_classes": ["healthy", "leaf-spot"],
        "ready": True,
    }


def test_knowledge_base_covers_every_model_class(
    knn_model: KnnPcaModel,
    cnn_model: CnnModel,
    mobilenetv2_model: MobileNetV2Model,
) -> None:
    knowledge = load_disease_knowledge_base()

    for model in (knn_model, cnn_model, mobilenetv2_model):
        assert set(model.metadata.supported_classes) <= knowledge.keys()


def test_saved_knn_pipeline_predicts_through_api(knn_model: KnnPcaModel) -> None:
    client = TestClient(create_app(knn_model))

    info_response = client.get("/api/v1/model/info")
    prediction_response = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.png", png_image(), "image/png")},
    )

    assert info_response.status_code == 200
    assert info_response.json()["architecture"] == "KNN + PCA"
    assert len(info_response.json()["supported_classes"]) == 38
    assert prediction_response.status_code == 200
    prediction = prediction_response.json()
    assert prediction["class_name"] in info_response.json()["supported_classes"]
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["model_version"] == info_response.json()["version"]
    assert prediction["disease_information"]["class_name"] == prediction["class_name"]


def test_saved_cnn_predicts_through_api(
    knn_model: KnnPcaModel, cnn_model: CnnModel
) -> None:
    client = TestClient(create_app(models={"knn": knn_model, "cnn": cnn_model}))

    info_response = client.get("/api/v1/model/info?model_id=cnn")
    prediction_response = client.post(
        "/api/v1/predict",
        data={"model_id": "cnn"},
        files={"file": ("leaf.png", png_image(), "image/png")},
    )

    assert info_response.status_code == 200
    assert info_response.json()["model_id"] == "cnn"
    assert info_response.json()["architecture"] == "Custom CNN"
    assert len(info_response.json()["supported_classes"]) == 38
    assert prediction_response.status_code == 200
    prediction = prediction_response.json()
    assert prediction["class_name"] in info_response.json()["supported_classes"]
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["model_version"] == info_response.json()["version"]
    assert prediction["disease_information"]["class_name"] == prediction["class_name"]


def test_model_list_includes_all_configured_models(
    knn_model: KnnPcaModel,
    cnn_model: CnnModel,
    mobilenetv2_model: MobileNetV2Model,
) -> None:
    response = TestClient(
        create_app(
            models={
                "knn": knn_model,
                "cnn": cnn_model,
                "mobilenet": mobilenetv2_model,
            }
        )
    ).get("/api/v1/models")

    assert response.status_code == 200
    assert [model["model_id"] for model in response.json()["models"]] == [
        "knn",
        "cnn",
        "mobilenet",
    ]


def test_default_registry_includes_mobilenetv2() -> None:
    response = TestClient(app).get("/api/v1/models")

    assert response.status_code == 200
    assert [model["model_id"] for model in response.json()["models"]] == [
        "knn",
        "cnn",
        "mobilenet",
    ]


def test_saved_mobilenetv2_predicts_through_api(
    mobilenetv2_model: MobileNetV2Model,
) -> None:
    client = TestClient(create_app(models={"mobilenet": mobilenetv2_model}))

    info_response = client.get("/api/v1/model/info?model_id=mobilenet")
    prediction_response = client.post(
        "/api/v1/predict",
        data={"model_id": "mobilenet"},
        files={"file": ("leaf.png", png_image(), "image/png")},
    )
    batch_response = client.post(
        "/api/v1/predict/batch",
        data={"model_id": "mobilenet"},
        files=[("files", ("leaf.png", png_image(), "image/png"))],
    )

    assert info_response.status_code == 200
    assert info_response.json()["model_id"] == "mobilenet"
    assert info_response.json()["architecture"] == "MobileNetV2"
    assert len(info_response.json()["supported_classes"]) == 38
    assert prediction_response.status_code == 200
    prediction = prediction_response.json()
    assert prediction["class_name"] in info_response.json()["supported_classes"]
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["model_version"] == info_response.json()["version"]
    assert prediction["disease_information"]["class_name"] == prediction["class_name"]
    assert batch_response.status_code == 200
    batch_prediction = batch_response.json()["predictions"][0]
    assert batch_prediction["class_name"] in info_response.json()["supported_classes"]
    assert batch_prediction["disease_information"]["class_name"] == batch_prediction["class_name"]


def test_single_prediction_accepts_an_image() -> None:
    response = TestClient(create_app(StubModel())).post(
        "/api/v1/predict",
        files={"file": ("leaf.png", png_image(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "class_name": "healthy",
        "confidence": 0.98,
        "model_version": "test-1",
        "disease_information": None,
    }


def test_batch_prediction_returns_one_result_per_image() -> None:
    response = TestClient(create_app(StubModel())).post(
        "/api/v1/predict/batch",
        files=[
            ("files", ("first.png", png_image(), "image/png")),
            ("files", ("second.png", png_image(), "image/png")),
        ],
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert all(
        prediction["disease_information"] is None
        for prediction in response.json()["predictions"]
    )


def test_unconfigured_model_does_not_return_a_fake_prediction() -> None:
    response = TestClient(create_app(UnconfiguredModel())).post(
        "/api/v1/predict",
        files={"file": ("leaf.png", png_image(), "image/png")},
    )

    assert response.status_code == 503


def test_prediction_rejects_non_image_content() -> None:
    response = TestClient(create_app(StubModel())).post(
        "/api/v1/predict",
        files={"file": ("leaf.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_prediction_rejects_oversized_images() -> None:
    response = TestClient(create_app(StubModel())).post(
        "/api/v1/predict",
        files={"file": ("leaf.png", b"x" * (MAX_IMAGE_BYTES + 1), "image/png")},
    )

    assert response.status_code == 413


def test_batch_rejects_too_many_files() -> None:
    response = TestClient(create_app(StubModel())).post(
        "/api/v1/predict/batch",
        files=[
            ("files", (f"{index}.png", png_image(), "image/png"))
            for index in range(MAX_BATCH_FILES + 1)
        ],
    )

    assert response.status_code == 413
