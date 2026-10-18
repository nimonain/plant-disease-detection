# Plant Leaf Detection frontend

React + TypeScript interface built with Vite. It uses the FastAPI prediction
API for readiness, model metadata, and single or batch image classification
(up to five images per batch).
The `/about` page describes the project, prediction workflow, available models,
image limits, technology, and data handling.
First-time visitors to the detector are guided through its main controls; the
tour can be restarted at any time with the **Tour** button in the header.

## Local run

Start the backend on `http://127.0.0.1:8000`, then from this directory:

```powershell
npm ci
Copy-Item .env.example .env
npm run dev
```

Open `http://127.0.0.1:5173`. By default the browser requests `/api/...` on the
same origin and Vite proxies these requests to the backend. Set `VITE_API_URL`
to override the API origin when needed.

## Production build and Render

```powershell
npm ci
npm run build
```

The output directory is `dist`. For Render, use the `frontend` root directory,
build command `npm ci && npm run build`, and publish directory `dist`. Set
`VITE_API_URL` to the deployed backend HTTPS origin (no trailing slash). The
Render Blueprint includes a rewrite from `/*` to `/index.html`; configure the
backend's `CORS_ORIGINS` to include this static site's exact HTTPS origin.
