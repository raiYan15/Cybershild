# CYBERSHIELD – AI Powered Cyberbullying Detection System

Production-style full-stack app with React + Express + MongoDB + FastAPI ML inference.

## Project Structure

- `frontend/` – React + Vite + Tailwind + Framer Motion + Recharts
- `backend/` – Express API with MongoDB storage and analytics
- `ml_api/` – FastAPI inference service source used by the backend (auto-started by backend)

## Environment Variables

Root `.env` is included with defaults:

- `PORT`
- `MONGODB_URI`
- `ML_API_URL`
- `VITE_API_URL`

You can also copy:

- `backend/.env.example`
- `frontend/.env.example`

## Run Locally

### 1) Backend API (includes ML API runtime)

From `backend/`:

- Install dependencies
- Ensure Python + `ml_api/requirements.txt` dependencies are installed
- Start server on port `5000` (backend auto-starts embedded ML API on `ML_INTERNAL_PORT`)

### 2) Frontend

From `frontend/`:

- Install dependencies
- Start Vite dev server on port `5173`

## API Endpoints

### Backend (`/api`)

- `POST /analyze`
- `POST /save`
- `GET /analytics`

### ML API (served internally by backend-managed process)

- `POST /predict`
- `GET /health`

## Notes on Pickle Integration

Uploaded files:

- `model (1).pkl`
- `vectorizer (1).pkl`

have been integrated into:

- `ml_api/model.pkl`
- `ml_api/vectorizer.pkl`

The FastAPI loader also supports both standard and `(1)` filenames.
