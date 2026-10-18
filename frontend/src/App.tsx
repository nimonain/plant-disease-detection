import { useEffect, useRef, useState } from "react";

type ModelInfo = {
  model_id: string;
  architecture: string;
  version: string;
  supported_classes: string[];
  ready: boolean;
};

type Prediction = {
  class_name: string;
  confidence: number;
  model_version: string;
  disease_information?: DiseaseInformation | null;
};

type DiseaseInformation = {
  class_name: string;
  crop: string;
  disease_name: string;
  status: string;
  cause: {
    type: string;
    organism: string | null;
  };
  description: string;
  symptoms: string[];
  prevention: string[];
  recommended_actions: string[];
  severity: string;
  spread_risk: string;
  treatment_note: string;
  disclaimer: string;
};

type Health = {
  status: string;
  service_ready: boolean;
  model_ready: boolean;
};

type ResultItem = {
  file: File;
  prediction: Prediction;
};

type ModelsResponse = {
  models: ModelInfo[];
};

const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_BATCH_FILES = 5;
const STATUS_TIMEOUT_MS = 5000;
const PREDICTION_TIMEOUT_MS = 60000;
const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const API_BASE_URL = (import.meta.env.VITE_API_URL ?? "").trim().replace(/\/+$/, "");
const TOUR_COMPLETED_KEY = "plant-disease-detection-tour-completed";
const TOUR_STEPS = [
  {
    target: "settings",
    title: "Choose how to analyze",
    description: "Use Single for one leaf or Batch for up to five images. If multiple models are available, choose one here.",
  },
  {
    target: "upload",
    title: "Add a leaf image",
    description: "Drop an image here or browse your files. JPEG, PNG, WEBP, and BMP images up to 10 MB are supported.",
  },
  {
    target: "camera",
    title: "Take a photo instead",
    description: "Open the camera to capture a fresh leaf photo. Your browser may ask for camera permission.",
  },
  {
    target: "analyze",
    title: "Run the prediction",
    description: "After selecting an image, choose Analyze image to get a predicted class and confidence score.",
  },
  {
    target: "model-status",
    title: "Review status and results",
    description: "This panel shows the active model and API status. Your prediction and any available disease guidance appear below the upload panel.",
  },
];

function apiUrl(path: string): string {
  if (!API_BASE_URL) return path;

  const url = new URL(path, `${API_BASE_URL}/`);
  if (import.meta.env.PROD && url.protocol !== "https:") {
    throw new Error("The production prediction API URL must use HTTPS.");
  }
  return url.toString();
}

async function fetchApi(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(apiUrl(path), { ...init, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error("The prediction API did not respond in time.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function readResponse<T>(response: Response): Promise<T> {
  const body = (await response.json().catch(() => null)) as
    | (T & { detail?: string })
    | null;
  if (!response.ok) {
    throw new Error(
      body?.detail ?? `Request failed with status ${response.status}.`,
    );
  }
  if (body === null) {
    throw new Error("The server returned an invalid response.");
  }
  return body;
}

function formatBytes(bytes: number): string {
  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function App() {
  const isAboutPage = window.location.pathname.replace(/\/+$/, "") === "/about";
  const [theme, setTheme] = useState<"light" | "dark">(
    () => window.localStorage.getItem("plant-leaf-detection-theme") === "dark"
      ? "dark"
      : "light",
  );
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [refreshSpinId, setRefreshSpinId] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState("knn");
  const [apiError, setApiError] = useState("");
  const [requestError, setRequestError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [tourStep, setTourStep] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const cameraButtonRef = useRef<HTMLButtonElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const model = models.find((available) => available.model_id === selectedModelId) ?? null;
  const singleFile = mode === "single" ? files[0] : undefined;
  const previewUrl = useObjectUrl(singleFile);

  useEffect(() => {
    window.localStorage.setItem("plant-leaf-detection-theme", theme);
  }, [theme]);

  useEffect(() => {
    document.title = isAboutPage
      ? "About | Plant disease detection"
      : "Plant disease detection";
  }, [isAboutPage]);

  useEffect(() => {
    if (!isAboutPage && window.localStorage.getItem(TOUR_COMPLETED_KEY) !== "true") {
      setTourStep(0);
    }
  }, [isAboutPage]);

  useEffect(() => {
    void refreshStatus();
  }, []);

  useEffect(() => {
    if (!cameraOpen || !videoRef.current || !cameraStreamRef.current) return;
    videoRef.current.srcObject = cameraStreamRef.current;
    void videoRef.current.play();
  }, [cameraOpen]);

  useEffect(() => {
    if (!cameraOpen) return;
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") closeCamera();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [cameraOpen]);

  useEffect(() => () => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  async function refreshStatus() {
    setApiError("");
    try {
      const [healthResponse, infoResponse] = await Promise.all([
        fetchApi("/api/v1/health", {}, STATUS_TIMEOUT_MS),
        fetchApi("/api/v1/models", {}, STATUS_TIMEOUT_MS),
      ]);
      const healthData = (await healthResponse.json()) as Health;
      if (!healthResponse.ok && healthResponse.status !== 503) {
        throw new Error(`Health check failed with status ${healthResponse.status}.`);
      }
      const modelData = await readResponse<ModelsResponse>(infoResponse);
      setHealth(healthData);
      setModels(modelData.models);
      setSelectedModelId((current) =>
        modelData.models.some((available) => available.model_id === current)
          ? current
          : modelData.models[0]?.model_id ?? "",
      );
    } catch (error) {
      setHealth(null);
      setModels([]);
      setApiError(
        error instanceof Error
          ? error.message
          : "Could not connect to the prediction API.",
      );
    }
  }

  function addFiles(fileList: FileList | File[]) {
    const selected = Array.from(fileList);
    setRequestError("");
    setResults([]);

    if (mode === "single" && selected.length > 1) {
      setRequestError("Choose one image in single-image mode.");
      return;
    }
    if (mode === "batch" && selected.length > MAX_BATCH_FILES) {
      setRequestError(`Choose no more than ${MAX_BATCH_FILES} images.`);
      return;
    }
    const invalidType = selected.find(
      (file) => !ACCEPTED_IMAGE_TYPES.includes(file.type),
    );
    if (invalidType) {
      setRequestError(`${invalidType.name} is not a supported image type.`);
      return;
    }
    const oversized = selected.find((file) => file.size > MAX_IMAGE_BYTES);
    if (oversized) {
      setRequestError(`${oversized.name} is larger than the 10 MB limit.`);
      return;
    }
    setFiles(selected);
  }

  async function classify() {
    if (files.length === 0 || loading) return;
    setLoading(true);
    setRequestError("");
    setResults([]);
    try {
      const form = new FormData();
      const isBatch = mode === "batch";
      for (const file of files) {
        form.append(isBatch ? "files" : "file", file);
      }
      form.append("model_id", selectedModelId);
      const response = await fetchApi(
        isBatch ? "/api/v1/predict/batch" : "/api/v1/predict",
        { method: "POST", body: form },
        PREDICTION_TIMEOUT_MS,
      );
      if (isBatch) {
        const body = await readResponse<{ predictions: Prediction[] }>(response);
        if (body.predictions.length !== files.length) {
          throw new Error("The API returned an unexpected number of predictions.");
        }
        setResults(
          files.map((file, index) => ({
            file,
            prediction: body.predictions[index],
          })),
        );
      } else {
        const prediction = await readResponse<Prediction>(response);
        setResults([{ file: files[0], prediction }]);
      }
    } catch (error) {
      setRequestError(
        error instanceof Error ? error.message : "Prediction request failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  function changeMode(nextMode: "single" | "batch") {
    setMode(nextMode);
    setFiles([]);
    setResults([]);
    setRequestError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDrop(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files.length) addFiles(event.dataTransfer.files);
  }

  async function openCamera() {
    setCameraError("");
    setCameraReady(false);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access is unavailable in this browser or page context.");
      return;
    }

    try {
      cameraStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" } },
      });
      setCameraOpen(true);
    } catch (error) {
      const name = error instanceof DOMException ? error.name : "";
      setCameraError(
        name === "NotAllowedError" || name === "SecurityError"
          ? "Camera permission was denied. Allow camera access in your browser settings and try again."
          : name === "NotFoundError"
            ? "No camera was found on this device."
            : name === "NotReadableError"
              ? "The camera is already in use by another application."
              : "Could not start the camera. Check its connection and browser permissions.",
      );
    }
  }

  function closeCamera() {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    setCameraOpen(false);
    setCameraReady(false);
    cameraButtonRef.current?.focus();
  }

  async function capturePhoto() {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      setCameraError("The camera is not ready yet. Wait for the preview and try again.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setCameraError("Could not capture an image from the camera.");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (capturedBlob) => capturedBlob
            ? resolve(capturedBlob)
            : reject(new Error("The browser could not encode the captured image.")),
          "image/jpeg",
          0.9,
        );
      });
      const file = new File(
        [blob],
        `camera-${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`,
        { type: "image/jpeg" },
      );
      if (file.size > MAX_IMAGE_BYTES) {
        setCameraError("The captured image is larger than the 10 MB limit.");
        return;
      }
      if (mode === "batch" && files.length >= MAX_BATCH_FILES) {
        setCameraError(`Remove an image before capturing; batches are limited to ${MAX_BATCH_FILES} images.`);
        return;
      }

      setFiles((current) => mode === "batch" ? [...current, file] : [file]);
      setResults([]);
      setRequestError("");
      closeCamera();
    } catch (error) {
      setCameraError(
        error instanceof Error ? error.message : "Could not capture a camera image.",
      );
    }
  }

  function finishTour() {
    window.localStorage.setItem(TOUR_COMPLETED_KEY, "true");
    setTourStep(null);
  }

  return (
    <div
      className="app-shell"
      data-theme={theme}
      onPointerMove={(event) => {
        if (event.pointerType !== "mouse") return;

        const bounds = event.currentTarget.getBoundingClientRect();
        event.currentTarget.style.setProperty(
          "--cursor-x",
          `${event.clientX - bounds.left}px`,
        );
        event.currentTarget.style.setProperty(
          "--cursor-y",
          `${event.clientY - bounds.top}px`,
        );
      }}
    >
      <header className="topbar">
        <a className="brand" href="/" aria-label="Plant disease detection home">
          <span className="brand-mark" aria-hidden="true">
            <LeafIcon />
          </span>
          <span>Plant disease detection</span>
        </a>
        <div className="topbar-right">
          <a
            className="about-link"
            href={isAboutPage ? "/" : "/about"}
            aria-current={isAboutPage ? "page" : undefined}
          >
            {isAboutPage ? "Detection" : "About"}
          </a>
          <button
            className="icon-button theme-toggle"
            type="button"
            onClick={() => setTheme((current) => current === "light" ? "dark" : "light")}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            <ThemeIcon theme={theme} />
          </button>
          {!isAboutPage && (
            <button
              className="tour-launch"
              type="button"
              onClick={() => setTourStep(0)}
            >
              <InfoIcon />
              <span>Tour</span>
            </button>
          )}
          <span className={`connection-dot ${health?.service_ready ? "online" : ""}`} />
          <span className="connection-label">
            {apiError ? "API offline" : "API online"}
          </span>
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              setRefreshSpinId((current) => current + 1);
              setRefreshing(true);
              void refreshStatus();
            }}
            aria-label="Refresh API status"
            title="Refresh API status"
          >
            <span
              key={refreshSpinId}
              className={`refresh-icon${refreshing ? " spinning" : ""}`}
              onAnimationEnd={() => setRefreshing(false)}
            >
              <RefreshIcon />
            </span>
          </button>
        </div>
      </header>

      <main>
        {isAboutPage ? <AboutPage /> : (
          <>
        {/*
        <section className="intro">
          <div className="eyebrow"><span /> PLANT HEALTH, AT A GLANCE</div>
          <h1>A closer look at<br />your <span>leaves.</span></h1>
          <p className="intro-copy">
            Upload a leaf image and let the model help identify what’s growing
            well — and what might need attention.
          </p>
        </section>
        */}

        <section className="workspace" aria-label="Leaf image classification">
          <div className="upload-panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">UPLOAD</span>
                <h2>Choose your images</h2>
              </div>
              <div className="upload-controls" data-tour="settings">
                <div className="mode-switch" role="group" aria-label="Upload mode">
                  <button
                    type="button"
                    aria-pressed={mode === "single"}
                    className={mode === "single" ? "selected" : ""}
                    onClick={() => changeMode("single")}
                  >
                    Single
                  </button>
                  <button
                    type="button"
                    aria-pressed={mode === "batch"}
                    className={mode === "batch" ? "selected" : ""}
                    onClick={() => changeMode("batch")}
                  >
                    Batch
                  </button>
                </div>
                {models.length > 1 && (
                  <ModelPicker
                    models={models}
                    selectedModelId={selectedModelId}
                    disabled={loading}
                    onChange={(modelId) => {
                      setSelectedModelId(modelId);
                      setResults([]);
                      setRequestError("");
                    }}
                  />
                )}
              </div>
            </div>

            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept={ACCEPTED_IMAGE_TYPES.join(",")}
              multiple={mode === "batch"}
              onChange={(event) => {
                if (event.target.files) addFiles(event.target.files);
                event.target.value = "";
              }}
            />
            <div
              className={`drop-zone ${dragging ? "dragging" : ""} ${singleFile ? "has-preview" : ""}`}
              data-tour="upload"
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  inputRef.current?.click();
                }
              }}
              role="button"
              tabIndex={0}
              aria-label={mode === "batch" ? "Choose image files" : "Choose an image"}
            >
              {singleFile && previewUrl ? (
                <div className="preview-wrap">
                  <img src={previewUrl} alt="Selected leaf preview" />
                  <div className="preview-caption">
                    <span className="preview-file-icon"><ImageIcon /></span>
                    <span className="preview-name">{singleFile.name}</span>
                    <span className="preview-size">{formatBytes(singleFile.size)}</span>
                  </div>
                  <span className="change-photo">Change image</span>
                </div>
              ) : (
                <div className="drop-content">
                  <span className="upload-icon"><UploadIcon /></span>
                  <strong>
                    {mode === "batch"
                      ? "Drop your images here"
                      : "Drop your image here"}
                  </strong>
                  <span className="drop-or">or</span>
                  <span className="browse-button">Browse files <span>↗</span></span>
                  <span className="file-hint">
                    {mode === "batch"
                      ? `Up to ${MAX_BATCH_FILES} images · 10 MB each`
                      : "PNG, JPG, WEBP · up to 10 MB"}
                  </span>
                </div>
              )}
            </div>
            <div className="camera-actions">
              <span>Or capture a fresh leaf photo</span>
              <button
                ref={cameraButtonRef}
                className="camera-button"
                data-tour="camera"
                type="button"
                onClick={() => void openCamera()}
                disabled={loading}
              >
                <CameraIcon />
                Open camera
              </button>
            </div>
            {cameraError && (
              <div className="error-message camera-error" role="alert">{cameraError}</div>
            )}

            {mode === "batch" && files.length > 0 && (
              <div className="file-list">
                <div className="file-list-heading">
                  <span>{files.length} image{files.length === 1 ? "" : "s"} selected</span>
                  <button type="button" onClick={() => setFiles([])}>Clear all</button>
                </div>
                {files.map((file, index) => (
                  <div className="file-row" key={`${file.name}-${file.lastModified}-${index}`}>
                    <span className="file-row-icon"><ImageIcon /></span>
                    <span className="file-row-name">{file.name}</span>
                    <span className="file-row-size">{formatBytes(file.size)}</span>
                    <button
                      type="button"
                      className="remove-file"
                      aria-label={`Remove ${file.name}`}
                      onClick={() => setFiles((current) => current.filter((_, i) => i !== index))}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}

            {requestError && <div className="error-message" role="alert">{requestError}</div>}

            <button
              className="classify-button"
              data-tour="analyze"
              type="button"
              disabled={!files.length || loading}
              onClick={() => void classify()}
            >
              {loading ? (
                <><span className="spinner" /> Analyzing leaf{mode === "batch" ? "s" : ""}…</>
              ) : (
                <>Analyze {mode === "batch" && files.length > 1 ? `${files.length} images` : "image"} <span>↗</span></>
              )}
            </button>
            <p className="privacy-note"><LockIcon /> Images are processed by your connected model API.</p>
          </div>

          <aside className="insight-panel" data-tour="model-status">
            <div className="panel-heading insight-heading">
              <div>
                <span className="step-label">MODEL</span>
                <h2>Model status</h2>
              </div>
              <div className="model-controls">
                <span className={`status-pill ${model?.ready ? "ready" : ""}`}>
                  <span />{model?.ready ? "Ready" : apiError ? "Offline" : "Not ready"}
                </span>
              </div>
            </div>

            {apiError ? (
              <div className="status-empty">
                <span className="status-symbol warning"><SignalIcon /></span>
                <strong>Can’t reach the API</strong>
                <p>{apiError}</p>
                <button type="button" onClick={() => void refreshStatus()}>Try again <span>↗</span></button>
              </div>
            ) : model ? (
              <>
                <div className="model-card">
                  <div className="model-card-top">
                    <span className="model-glyph"><LeafIcon /></span>
                    <span className="model-tag">ACTIVE MODEL</span>
                  </div>
                  <h3>{model.architecture}</h3>
                  <p>Version {model.version}</p>
                </div>
                {!model.supported_classes.length && (
                  <div className="not-configured">
                    <span className="not-configured-icon"><InfoIcon /></span>
                    <div>
                      <strong>Model not configured</strong>
                      <p>Connect trained weights to enable predictions.</p>
                    </div>
                  </div>
                )}
                <div className="api-footnote">
                  <span className={`connection-dot ${health?.service_ready ? "online" : ""}`} />
                  <span>{health?.service_ready ? "Prediction service connected" : "Prediction service unavailable"}</span>
                </div>
              </>
            ) : (
              <div className="loading-model"><span className="spinner dark" /> Checking model status…</div>
            )}
          </aside>
        </section>

        {results.length > 0 && (
          <section className="results-section" aria-live="polite">
            <div className="results-heading">
              <div>
                <span className="step-label">RESULTS</span>
                <h2>Analysis complete</h2>
              </div>
              <span className="result-count">{results.length} result{results.length === 1 ? "" : "s"}</span>
            </div>
            <div className="results-grid">
              {results.map(({ file, prediction }, index) => (
                <article className="result-card" key={`${file.name}-${index}`}>
                  <div className="result-file"><ImageIcon /><span>{file.name}</span></div>
                  <div className="result-class">
                    {prediction.disease_information?.disease_name ??
                      prediction.class_name.replaceAll("_", " ")}
                  </div>
                  <div className="confidence-line">
                    <span>Confidence</span>
                    <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="confidence-track">
                    <span style={{ width: `${Math.max(0, Math.min(100, prediction.confidence * 100))}%` }} />
                  </div>
                  {prediction.disease_information &&
                    prediction.disease_information.status !== "healthy" && (
                      <div className="disease-guidance">
                        <div className="guidance-status">
                          {prediction.disease_information.crop} · {prediction.disease_information.status}
                        </div>
                        <p>{prediction.disease_information.description}</p>
                        <div className="guidance-detail">
                          <strong>Cause</strong>
                          <span>
                            {prediction.disease_information.cause.type}
                            {prediction.disease_information.cause.organism
                              ? ` · ${prediction.disease_information.cause.organism}`
                              : ""}
                          </span>
                        </div>
                        <GuidanceList
                          title="What to do"
                          items={prediction.disease_information.recommended_actions}
                        />
                        <GuidanceList
                          title="Prevention"
                          items={prediction.disease_information.prevention}
                        />
                        <div className="guidance-note">
                          {prediction.disease_information.treatment_note}
                        </div>
                        <div className="guidance-disclaimer">
                          {prediction.disease_information.disclaimer}
                        </div>
                      </div>
                    )}
                </article>
              ))}
            </div>
          </section>
        )}

        <footer>
          <span>made by group 29 with love</span>
        </footer>
          </>
        )}
      </main>
      {cameraOpen && (
        <div
          className="camera-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) closeCamera();
          }}
        >
          <section
            className="camera-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="camera-title"
          >
            <div className="camera-dialog-heading">
              <div>
                <span className="step-label">CAMERA CAPTURE</span>
                <h2 id="camera-title">Take a leaf photo</h2>
              </div>
              <button
                className="camera-close"
                type="button"
                onClick={closeCamera}
                aria-label="Close camera"
              >
                ×
              </button>
            </div>
            <div className="camera-preview">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                onCanPlay={() => setCameraReady(true)}
                aria-label="Live camera preview"
              />
              {!cameraReady && (
                <div className="camera-loading">
                  <span className="spinner" />
                  Starting camera…
                </div>
              )}
            </div>
            {cameraError && <div className="error-message" role="alert">{cameraError}</div>}
            <div className="camera-dialog-actions">
              <button className="camera-cancel" type="button" onClick={closeCamera}>
                Cancel
              </button>
              <button
                className="classify-button camera-capture"
                type="button"
                onClick={() => void capturePhoto()}
                disabled={!cameraReady}
              >
                <CameraIcon />
                Capture photo
              </button>
            </div>
          </section>
        </div>
      )}
      {tourStep !== null && !isAboutPage && (
        <GuidedTour
          step={tourStep}
          onStepChange={setTourStep}
          onFinish={finishTour}
        />
      )}
    </div>
  );
}

function GuidedTour({
  step,
  onStepChange,
  onFinish,
}: {
  step: number;
  onStepChange: (step: number) => void;
  onFinish: () => void;
}) {
  const [targetRect, setTargetRect] = useState<{
    top: number;
    left: number;
    width: number;
    height: number;
  } | null>(null);
  const [cardPosition, setCardPosition] = useState<{ top: number; left: number } | null>(null);
  const cardRef = useRef<HTMLElement>(null);
  const nextButtonRef = useRef<HTMLButtonElement>(null);
  const currentStep = TOUR_STEPS[step];

  useEffect(() => {
    const target = document.querySelector<HTMLElement>(
      `[data-tour="${currentStep.target}"]`,
    );
    if (!target) {
      setTargetRect(null);
      return;
    }
    const targetElement: HTMLElement = target;

    setTargetRect(null);
    setCardPosition(null);
    targetElement.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });

    function updatePosition() {
      const targetBounds = targetElement.getBoundingClientRect();
      const padding = 7;
      setTargetRect({
        top: targetBounds.top - padding,
        left: targetBounds.left - padding,
        width: targetBounds.width + padding * 2,
        height: targetBounds.height + padding * 2,
      });

      const cardBounds = cardRef.current?.getBoundingClientRect();
      const cardHeight = cardBounds?.height ?? 210;
      const cardWidth = Math.min(380, window.innerWidth - 32);
      const top =
        targetBounds.bottom + cardHeight + 16 <= window.innerHeight - 16
          ? targetBounds.bottom + 16
          : Math.max(16, targetBounds.top - cardHeight - 16);
      const left = Math.max(
        16,
        Math.min(
          targetBounds.left + targetBounds.width / 2 - cardWidth / 2,
          window.innerWidth - cardWidth - 16,
        ),
      );
      setCardPosition({ top, left });
    }

    const frame = window.requestAnimationFrame(updatePosition);
    const settleTimer = window.setTimeout(updatePosition, 350);
    window.addEventListener("scroll", updatePosition, true);
    window.addEventListener("resize", updatePosition);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(settleTimer);
      window.removeEventListener("scroll", updatePosition, true);
      window.removeEventListener("resize", updatePosition);
    };
  }, [currentStep.target, step]);

  useEffect(() => {
    nextButtonRef.current?.focus();
  }, [step]);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") onFinish();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onFinish]);

  function handleKeyDown(event: React.KeyboardEvent<HTMLElement>) {
    if (event.key !== "Tab") return;
    const focusable = cardRef.current?.querySelectorAll<HTMLElement>(
      "button:not(:disabled)",
    );
    if (!focusable?.length) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div className="guided-tour">
      <div className="tour-shade" aria-hidden="true" />
      {targetRect && (
        <div
          className="tour-spotlight"
          aria-hidden="true"
          style={{
            top: targetRect.top,
            left: targetRect.left,
            width: targetRect.width,
            height: targetRect.height,
          }}
        />
      )}
      <section
        ref={cardRef}
        className="tour-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-title"
        aria-describedby="tour-description"
        style={{
          top: cardPosition?.top ?? "50%",
          left: cardPosition?.left ?? "50%",
          visibility: cardPosition ? "visible" : "hidden",
          transform: cardPosition ? undefined : "translate(-50%, -50%)",
        }}
        onKeyDown={handleKeyDown}
      >
        <div className="tour-card-top">
          <span className="tour-counter">STEP {step + 1} OF {TOUR_STEPS.length}</span>
          <button className="tour-skip" type="button" onClick={onFinish}>Skip tour</button>
        </div>
        <h2 id="tour-title">{currentStep.title}</h2>
        <p id="tour-description">{currentStep.description}</p>
        <div className="tour-progress" aria-hidden="true">
          {TOUR_STEPS.map((item, index) => (
            <span className={index <= step ? "active" : ""} key={item.target} />
          ))}
        </div>
        <div className="tour-actions">
          {step > 0 && (
            <button
              className="tour-back"
              type="button"
              onClick={() => onStepChange(step - 1)}
            >
              Back
            </button>
          )}
          <button
            ref={nextButtonRef}
            className="tour-next"
            type="button"
            onClick={() => step === TOUR_STEPS.length - 1 ? onFinish() : onStepChange(step + 1)}
          >
            {step === TOUR_STEPS.length - 1 ? "Finish" : "Next"}
          </button>
        </div>
      </section>
    </div>
  );
}

function AboutPage() {
  return (
    <>
      <section className="about-hero">
        <div className="eyebrow"><span /> ABOUT THE PROJECT</div>
        <h1>Healthier plants,<br /><span>one leaf at a time.</span></h1>
        <p className="intro-copy">
          Plant disease detection helps you explore what may be affecting a
          plant from a photo of its leaf, using image-classification models and
          practical disease guidance.
        </p>
        <a className="about-button" href="/">Try the detector <span aria-hidden="true">→</span></a>
      </section>

      <section className="about-grid" aria-label="Project information">
        <article className="about-card about-card-wide">
          <span className="section-label">HOW IT WORKS</span>
          <h2>From leaf photo to useful insight</h2>
          <ol className="about-steps">
            <li><strong>Add a photo.</strong> Upload an image or capture one with your device camera.</li>
            <li><strong>Choose a model.</strong> Compare predictions from the available classifiers.</li>
            <li><strong>Review the result.</strong> See the predicted class, confidence, and available disease information and recommended actions.</li>
          </ol>
        </article>

        <article className="about-card">
          <span className="section-label">CLASSIFIERS</span>
          <h2>Three models to explore</h2>
          <ul className="about-models">
            <li><strong>KNN + PCA</strong><span>A feature-based pipeline using dimensionality reduction and K-nearest neighbors.</span></li>
            <li><strong>Custom CNN</strong><span>A convolutional neural network for leaf-image classification.</span></li>
            <li><strong>MobileNetV2</strong><span>A convolutional neural network architecture adapted for image classification.</span></li>
          </ul>
        </article>

        <article className="about-card">
          <span className="section-label">IMAGE INPUT</span>
          <h2>Upload or capture</h2>
          <ul className="about-facts">
            <li>JPEG, PNG, WEBP, and BMP images</li>
            <li>Up to 10 MB per image</li>
            <li>Single-image or batch predictions, with up to five images per batch</li>
            <li>Camera capture where supported by your browser</li>
          </ul>
        </article>

        <article className="about-card">
          <span className="section-label">TECHNOLOGY</span>
          <h2>Built for image inference</h2>
          <p>
            The interface is built with React, TypeScript, and Vite. A FastAPI
            service runs the prediction models with TensorFlow/Keras and
            scikit-learn, and provides the disease-guidance API.
          </p>
        </article>

        <article className="about-card">
          <span className="section-label">DATA &amp; LIMITATIONS</span>
          <h2>Use results as a guide</h2>
          <p>
            Images are sent to the configured prediction API for processing.
            The current backend does not require an account or database and
            does not use persistent upload storage. Predictions are
            classifications, not a substitute for expert diagnosis; image
            quality and model limitations can affect results.
          </p>
        </article>

        <article className="about-card about-card-wide about-resources">
          <span className="section-label">DEVELOPER RESOURCES</span>
          <h2>Explore the API</h2>
          <p>
            The API provides readiness and model information alongside
            single-image and batch prediction endpoints. When running locally,
            interactive API documentation is available at{" "}
            <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
              127.0.0.1:8000/docs
            </a>.
          </p>
        </article>
      </section>

      <footer>
        <span>Plant disease detection</span>
      </footer>
    </>
  );
}

function GuidanceList({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="guidance-list">
      <strong>{title}</strong>
      <ul>
        {items.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  );
}

function ModelPicker({
  models,
  selectedModelId,
  disabled,
  onChange,
}: {
  models: ModelInfo[];
  selectedModelId: string;
  disabled: boolean;
  onChange: (modelId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const pickerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const selectedIndex = models.findIndex((item) => item.model_id === selectedModelId);
  const selectedModel = models[selectedIndex];

  useEffect(() => {
    if (!open) return;
    function closeWhenOutside(event: PointerEvent) {
      if (!pickerRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    }
    document.addEventListener("pointerdown", closeWhenOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeWhenOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  function focusOption(index: number) {
    const wrappedIndex = ((index % models.length) + models.length) % models.length;
    optionRefs.current[wrappedIndex]?.focus();
  }

  return (
    <div className={`model-picker ${open ? "open" : ""} ${disabled ? "disabled" : ""}`} ref={pickerRef}>
      <button
        ref={triggerRef}
        className="model-picker-trigger"
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`Prediction model: ${selectedModel?.architecture ?? "none selected"}`}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            setOpen(true);
            requestAnimationFrame(() => {
              const initialIndex = selectedIndex < 0 ? 0 : selectedIndex;
              focusOption(event.key === "ArrowDown" ? initialIndex : initialIndex - 1);
            });
          }
        }}
      >
        <span className="model-picker-icon"><LeafIcon /></span>
        <span className="model-picker-label">{selectedModel?.architecture ?? "Choose model"}</span>
        <svg className="model-picker-chevron" viewBox="0 0 12 12" fill="none" aria-hidden="true">
          <path d="m3 4.5 3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="model-picker-menu" role="listbox" aria-label="Prediction model">
          {models.map((available, index) => (
            <button
              ref={(element) => { optionRefs.current[index] = element; }}
              className={`model-picker-option ${available.model_id === selectedModelId ? "selected" : ""}`}
              id={`model-option-${available.model_id}`}
              key={available.model_id}
              type="button"
              role="option"
              aria-selected={available.model_id === selectedModelId}
              onClick={() => {
                onChange(available.model_id);
                setOpen(false);
                triggerRef.current?.focus();
              }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  focusOption(index + 1);
                } else if (event.key === "ArrowUp") {
                  event.preventDefault();
                  focusOption(index - 1);
                } else if (event.key === "Home") {
                  event.preventDefault();
                  focusOption(0);
                } else if (event.key === "End") {
                  event.preventDefault();
                  focusOption(models.length - 1);
                } else if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onChange(available.model_id);
                  setOpen(false);
                  triggerRef.current?.focus();
                }
              }}
            >
              <span>{available.architecture}</span>
              {available.model_id === selectedModelId && <span className="model-picker-check" aria-hidden="true">✓</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function useObjectUrl(file?: File): string | undefined {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!file) {
      setUrl(undefined);
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);
  return url;
}

function LeafIcon() {
  return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20.5 3.5C12 3.7 5.2 5.3 4 11.1c-.7 3.3 1.4 5.6 4.3 5.2 5.3-.7 9.9-6.5 12.2-12.8Z" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/><path d="M3.5 21c2.6-5.7 6-9.1 11.4-12.3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/></svg>;
}

function UploadIcon() {
  return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4.5A1.5 1.5 0 006.5 20h11a1.5 1.5 0 001.5-1.5V14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function CameraIcon() {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M6.3 5.2 7.4 3.7h5.2l1.1 1.5h2.1a1.7 1.7 0 011.7 1.7v7.2a1.7 1.7 0 01-1.7 1.7H4.2a1.7 1.7 0 01-1.7-1.7V6.9a1.7 1.7 0 011.7-1.7h2.1Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round"/><circle cx="10" cy="10.2" r="3" stroke="currentColor" strokeWidth="1.4"/></svg>;
}

function ImageIcon() {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="2.5" y="3" width="15" height="14" rx="2" stroke="currentColor" strokeWidth="1.4"/><circle cx="7" cy="7.5" r="1.5" stroke="currentColor" strokeWidth="1.4"/><path d="m3.5 14 4-4 3.1 3 2.4-2 3.5 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function ThemeIcon({ theme }: { theme: "light" | "dark" }) {
  return theme === "light"
    ? <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M16.7 12.2A7 7 0 017.8 3.3 7 7 0 1016.7 12.2Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
    : <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="10" cy="10" r="3.2" stroke="currentColor" strokeWidth="1.5"/><path d="M10 2v1.5m0 13V18m8-8h-1.5m-13 0H2m13.66-5.66-1.06 1.06m-9.2 9.2-1.06 1.06m11.32 0-1.06-1.06m-9.2-9.2L4.34 4.34" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>;
}

function RefreshIcon() {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M16.2 8A6.5 6.5 0 104.7 14.7M16.2 8V3.8M16.2 8H12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function LockIcon() {
  return <svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><rect x="3.3" y="7" width="9.4" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.2"/><path d="M5.3 7V4.8a2.7 2.7 0 015.4 0V7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>;
}

function SignalIcon() {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 6.5v4m0 3h.01M3.1 15.5 8.7 4.8a1.5 1.5 0 012.6 0l5.6 10.7a1.5 1.5 0 01-1.3 2.2H4.4a1.5 1.5 0 01-1.3-2.2Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function InfoIcon() {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="10" cy="10" r="7.2" stroke="currentColor" strokeWidth="1.4"/><path d="M10 9v4m0-7h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/></svg>;
}

export default App;
