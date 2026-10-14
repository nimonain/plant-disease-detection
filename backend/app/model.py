from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Protocol

import joblib
import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline

from app.features import ImageFeatureExtractor

if TYPE_CHECKING:
    import tensorflow as tf

MODEL_DIR = Path(__file__).with_name("models")
MODEL_PATH = MODEL_DIR / "knn_pipeline.joblib"
CNN_MODEL_PATH = MODEL_DIR / "cnn_model.keras"
CNN_CLASSES_PATH = MODEL_DIR / "cnn_class_names.json"
MOBILENETV2_MODEL_PATH = MODEL_DIR / "mobilenetv2_model.keras"
MOBILENETV2_CLASSES_PATH = MODEL_DIR / "mobilenetv2_class_names.json"


@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    version: str
    supported_classes: list[str]
    ready: bool


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float


class ModelNotReadyError(RuntimeError):
    """Raised when prediction is requested before a model is configured."""


class PredictionModel(Protocol):
    @property
    def metadata(self) -> ModelMetadata: ...

    def predict(self, image: Image.Image, /) -> Prediction: ...


class UnconfiguredModel:
    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            architecture="unconfigured",
            version="unconfigured",
            supported_classes=[],
            ready=False,
        )

    def predict(self, _: Image.Image) -> Prediction:
        raise ModelNotReadyError("No prediction model has been configured.")


class KnnPcaModel:
    def __init__(self, pipeline: Pipeline, model_path: Path) -> None:
        if not hasattr(pipeline, "classes_"):
            raise ValueError("The KNN pipeline has no fitted class labels.")
        if not hasattr(pipeline, "predict_proba"):
            raise ValueError("The KNN pipeline must provide predict_proba().")

        self._pipeline = pipeline
        self._classes = [str(label) for label in pipeline.classes_]
        self._lock = threading.Lock()
        artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        self._metadata = ModelMetadata(
            architecture="KNN + PCA",
            version=f"knn-pca-{artifact_hash}",
            supported_classes=self._classes,
            ready=True,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, image: Image.Image) -> Prediction:
        with TemporaryDirectory(prefix="leaf-predict-") as temporary_directory:
            image_path = Path(temporary_directory) / "leaf.png"
            image.convert("RGB").save(image_path, format="PNG")
            with self._lock:
                features = self._pipeline[:-1].transform([str(image_path)])
                classifier = self._pipeline.steps[-1][1]
                predicted = classifier.predict(features)[0]
                probabilities = classifier.predict_proba(features)[0]

        predicted_class = str(predicted)
        if predicted_class not in self._classes:
            raise ValueError("The KNN pipeline returned an unknown class label.")
        confidence = float(np.max(probabilities))
        return Prediction(class_name=predicted_class, confidence=confidence)


def load_knn_model(model_path: Path = MODEL_PATH) -> KnnPcaModel:
    if not model_path.is_file():
        raise FileNotFoundError(f"KNN model artifact was not found: {model_path}")

    # The notebook serialized this transformer as __main__.ImageFeatureExtractor.
    setattr(sys.modules["__main__"], "ImageFeatureExtractor", ImageFeatureExtractor)
    pipeline = joblib.load(model_path)

    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected a fitted sklearn Pipeline in {model_path}.")
    step_names = [name for name, _ in pipeline.steps]
    expected_steps = ["features", "scaler", "pca", "knn"]
    if step_names != expected_steps:
        raise ValueError(
            f"Unexpected KNN pipeline steps {step_names}; expected {expected_steps}."
        )
    return KnnPcaModel(pipeline, model_path)


class _KerasImageModel:
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: list[str],
        model_path: Path,
        *,
        architecture: str,
        version_prefix: str,
    ) -> None:
        output_shape = getattr(model, "output_shape", None)
        input_shape = getattr(model, "input_shape", None)
        if (
            not isinstance(output_shape, tuple)
            or len(output_shape) != 2
            or output_shape[-1] != len(class_names)
        ):
            raise ValueError(
                f"{architecture} output count does not match the supplied class-name list."
            )
        if input_shape != (None, 224, 224, 3):
            raise ValueError(
                f"Expected {architecture} input shape (None, 224, 224, 3), got {input_shape}."
            )

        self._model = model
        self._class_names = class_names
        self._lock = threading.Lock()
        artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        self._metadata = ModelMetadata(
            architecture=architecture,
            version=f"{version_prefix}-{artifact_hash}",
            supported_classes=class_names,
            ready=True,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, image: Image.Image) -> Prediction:
        import tensorflow as tf

        resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        image_array = tf.keras.utils.img_to_array(resized)
        batch = np.expand_dims(image_array, axis=0)
        with self._lock:
            probabilities = self._model.predict(batch, verbose=0)[0]

        if probabilities.shape != (len(self._class_names),):
            raise ValueError(
                "CNN prediction output count does not match the class-name list."
            )
        predicted_index = int(np.argmax(probabilities))
        return Prediction(
            class_name=self._class_names[predicted_index],
            confidence=float(probabilities[predicted_index]),
        )


class CnnModel(_KerasImageModel):
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: list[str],
        model_path: Path,
    ) -> None:
        super().__init__(
            model,
            class_names,
            model_path,
            architecture="Custom CNN",
            version_prefix="cnn",
        )


class MobileNetV2Model(_KerasImageModel):
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: list[str],
        model_path: Path,
    ) -> None:
        super().__init__(
            model,
            class_names,
            model_path,
            architecture="MobileNetV2",
            version_prefix="mobilenetv2",
        )

    def predict(self, image: Image.Image) -> Prediction:
        import tensorflow as tf

        resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        image_array = tf.keras.utils.img_to_array(resized)
        batch = np.expand_dims(image_array, axis=0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        with self._lock:
            probabilities = self._model.predict(batch, verbose=0)[0]

        if probabilities.shape != (len(self._class_names),):
            raise ValueError(
                "MobileNetV2 prediction output count does not match the class-name list."
            )
        predicted_index = int(np.argmax(probabilities))
        return Prediction(
            class_name=self._class_names[predicted_index],
            confidence=float(probabilities[predicted_index]),
        )


def _load_class_names(classes_path: Path, model_name: str) -> list[str]:
    class_names = json.loads(classes_path.read_text(encoding="utf-8"))
    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError(
            f"{model_name} class names in {classes_path} must be a non-empty list of unique strings."
        )
    return class_names


def load_cnn_model(
    model_path: Path = CNN_MODEL_PATH,
    classes_path: Path = CNN_CLASSES_PATH,
) -> CnnModel:
    if not model_path.is_file():
        raise FileNotFoundError(f"CNN model artifact was not found: {model_path}")
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"CNN class-name file was not found: {classes_path}"
        )

    class_names = _load_class_names(classes_path, "CNN")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    return CnnModel(model, class_names, model_path)


def load_mobilenetv2_model(
    model_path: Path = MOBILENETV2_MODEL_PATH,
    classes_path: Path = MOBILENETV2_CLASSES_PATH,
) -> MobileNetV2Model:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MobileNetV2 model artifact was not found: {model_path}"
        )
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"MobileNetV2 class-name file was not found: {classes_path}"
        )

    class_names = _load_class_names(classes_path, "MobileNetV2")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    return MobileNetV2Model(model, class_names, model_path)
