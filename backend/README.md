# Leaf Disease Prediction API

FastAPI backend for single-image and batch leaf-image classification. The
application loads the KNN + PCA pipeline, custom CNN, and MobileNetV2 from
`app/models/`, and disease guidance from `app/prevention/`. These artifacts
are included in this directory; no model download, database, or persistent
upload storage is needed.

## Local run

From the repository root:

```powershell
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs`.
`GET /health` and `GET /api/v1/health` return model readiness.

For tests, install `backend/requirements-dev.txt`, then from the repository
root run `Push-Location backend; python -m pytest tests; Pop-Location`.

## Render

The repository-root Blueprint configures this service with:

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`
- Environment variables: `APP_ENV=production` and `CORS_ORIGINS` containing
  the exact HTTPS origin(s) of the deployed frontend, comma-separated.

Use Python 3.12.10 (`PYTHON_VERSION`). See the root README for full Render
setup and post-deployment checks.

Individual uploads are limited to 10 MiB. Batch inference accepts up to five
images per request. Invalid image content returns `415`, oversized uploads
return `413`, and model loading errors fail application startup.
