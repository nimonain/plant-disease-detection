
# Plant Leaf Detection

Plant Leaf Detection is a React + TypeScript (Vite) frontend and a FastAPI
backend for leaf-image disease classification. The backend uses committed
KNN/PCA, CNN, and MobileNetV2 artifacts under `backend/app/models/`; no model
or dataset is downloaded at application startup. The API also loads its
disease guidance from `backend/app/prevention/`.

## Local setup

Use Python 3.12 for the backend (TensorFlow dependency compatibility) and
Node.js with npm for the frontend. From the repository root, in PowerShell:

```powershell
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to the backend during
development. API docs are at `http://127.0.0.1:8000/docs`, and readiness is
available at `http://127.0.0.1:8000/health`. Frontend production output is
created in `frontend/dist`.

To verify the production frontend build:

```powershell
cd frontend
npm ci
npm run build
```

To run backend tests from the repository root after installing
`backend/requirements-dev.txt`:

```powershell
python -m pip install -r backend\requirements-dev.txt
Push-Location backend
python -m pytest tests
Pop-Location
```

## Deploy to Render

The root [`render.yaml`](./render.yaml) defines a Python web service and a
static site. In Render, create a **Blueprint** from this GitHub repository and
review both services before applying. The backend service must start and become
healthy before the frontend can make successful API requests.

### Backend web service

- **Root Directory:** `backend`
- **Runtime:** Python
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`
- **Python:** `3.12.10` (set as `PYTHON_VERSION`)
- **Environment variables:**
  - `APP_ENV=production`
  - `CORS_ORIGINS=https://<your-render-static-site-host>`; enter the exact
    frontend origin (scheme and host only, no path or trailing slash). Add any
    custom frontend domain as another comma-separated origin.

The health endpoint returns `503` if model readiness is false. Model artifacts
are included in the repository; the web service does not require a database,
secret key, or external API credential.

### Frontend static site

- **Root Directory:** `frontend`
- **Build Command:** `npm ci && npm run build`
- **Publish Directory:** `dist`
- **Environment variable:** `VITE_API_URL=https://<your-render-backend-host>`
  (the deployed backend's HTTPS origin, without a trailing slash)
- **Rewrite:** `/*` to `/index.html`

The Vite build embeds `VITE_API_URL` into the public JavaScript bundle. It must
contain only the backend's public URL, never credentials or private API keys.
Render prompts for the two `sync: false` variables when you first create the
Blueprint. Enter the static-site HTTPS origin for `CORS_ORIGINS` and the
backend HTTPS origin for `VITE_API_URL`; use the hostnames assigned to these
services in Render. If the final service URLs differ from the values entered,
update the corresponding variable in that service's Dashboard settings. A
missing `CORS_ORIGINS` prevents the production backend from starting. Redeploy
the backend after changing CORS, and redeploy the frontend after changing its
build-time `VITE_API_URL`.

### Deployment and smoke-test checklist

1. Confirm both services have successful builds and the backend health check is
   passing.
2. Request `https://<backend-host>/health`; expect HTTP 200 and
   `"model_ready": true`.
3. Request `https://<backend-host>/api/v1/models`; expect the configured model
   list.
4. Open the static-site HTTPS URL, confirm the status shows the API online,
   then upload an image and run a prediction.
5. Test a direct frontend path such as `/anything`; it should serve the SPA
   entry page.
6. In browser developer tools, confirm API requests use HTTPS and that requests
   from the frontend origin succeed. Requests from an unconfigured origin
   should not receive an `Access-Control-Allow-Origin` response header.
7. Check Render service logs if startup or inference fails; do not put secrets
   in logs or public frontend environment variables.

Model artifacts are approximately 45 MB total for the three models loaded by
the API (KNN/PCA, CNN, and MobileNetV2). TensorFlow and all three models load
at process startup, so choose a Render instance with sufficient memory for
TensorFlow inference. Free-tier memory may be insufficient; verify startup and
prediction on the selected instance before relying on it. No dataset, database,
uploads directory, or persistent disk is required by the current application.
