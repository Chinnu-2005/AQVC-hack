## Quantum ML FTSE 100 Predictor

A production‑ready demo that forecasts intraday FTSE 100 movements using a Variational Quantum Classifier (VQC) with a modern React UI. The backend auto‑trains on a sliding 3‑year window and caches per‑date predictions for instant reloads.

### Features
- **Quantum ML core**: Qiskit VQC with stratified train/test split and standardized features
- **Intraday forecasts**: 08:00, 10:00, 12:00, 14:00, 16:00 predictions
- **Auto‑training**: Refreshes at most once per day for “today”
- **CSV caching**: Historical date responses are cached on disk and persist across restarts
- **Clean UI**: Responsive Home, About, and Dashboard pages

### Project Structure
```text
AQVC-hack/
  backend/
    logic/               # FastAPI app factory
    middleware/          # Error handling
    model/               # Predictor and pydantic schemas
    router/              # REST endpoints
    service/             # Model service + caching
    util/                # Settings, logging, exceptions
    models/              # Saved model + metadata (persisted)
    main.py              # FastAPI entrypoint
    requirements.txt
  frontend/
    src/                 # React app (Home, About, Dashboard)
    public/
    package.json
  README.md
```

### Tech Stack
- Backend: FastAPI, Qiskit, scikit‑learn, pandas, yfinance
- Frontend: React (CRA), Chart.js

## Getting Started

### Prerequisites
- Python 3.10+
- Node 18+

### 1) Backend setup (Windows PowerShell)
```powershell
cd backend
py -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Optional: configure environment (see .env section below)

# Start API
python main.py
```
The API starts at `http://localhost:8000`.

### 2) Frontend setup
```powershell
cd frontend
npm install
npm start
```
The dev server runs at `http://localhost:3000` and proxies to the backend (`package.json` proxy).

## Configuration (.env)
Create `backend/.env` to override defaults from `backend/util/config.py`.
```env
# Server
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO

# Model
RANDOM_SEED=42
FEATURE_DIM=8
QUANTUM_REPS=2
MODEL_DIR=models
MODEL_FILE=quantum_vqc.joblib
MODEL_METADATA_FILE=model_metadata.json

# Data & training
DEFAULT_LOOKBACK_PERIOD=10
DEFAULT_TEST_SIZE=0.2

# Cache
CACHE_DIR=cache
CACHE_FILE=predictions_cache.csv

# Optional database (currently unused)
MONGODB_URI=
```

## API Reference
Base URL: `http://localhost:8000`

- `GET /model/status` — Current model metadata (trained, accuracy, last_trained, etc.)
- `POST /model/train` — Train manually (uses sliding 3‑year window by default)
- `POST /model/predict` — Predict from a custom feature vector
- `POST /model/predict/latest` — Intraday predictions for today; auto‑trains at most once per day
- `POST /model/date?target_date=YYYY-MM-DD` — Intraday predictions for a specific date; uses cache when available

### Example: date prediction
```bash
curl -X POST "http://localhost:8000/model/date?target_date=2025-08-21"
```
Response includes `predictions`, `training_accuracy`, `training_period`, `actual_data` (if available), etc.

## Caching Behavior
- Historical date requests (`/model/date`) are cached in `cache/predictions_cache.csv` and served instantly on repeat, even after server restarts.
- Today’s endpoint (`/model/predict/latest`) checks if the model needs retraining based on `last_trained` and only retrains at most once per day.
- Clear a specific date by deleting its row from the CSV, or clear all by deleting the CSV file.

### Cache schema (columns)
`target_date, is_historical_date, training_period, training_accuracy, training_samples, predictions, base_feature_vector, actual_data, timestamp`

## Development Notes
- Saved model and metadata are stored in `backend/models/`.
- Training accuracy is computed on the test split and persisted in `model_metadata.json`.
- Frontend UI emphasizes performance and readability; the dashboard logic remains unchanged while Home/About use a modern hero/feature layout.

## Scripts
Backend
```bash
# From backend/
python main.py                  # start API
```
Frontend
```bash
# From frontend/
npm start                       # dev server
npm run build                   # production build
```

## Troubleshooting
- Port in use: stop existing processes on 8000/3000 or change ports.
- Qiskit install issues: ensure a clean virtual environment and compatible Python version.
- No predictions/empty features: verify market data availability (yfinance) and system time.
- Want a fresh run: delete `backend/cache/predictions_cache.csv` and files in `backend/models/`.

## License
MIT (or your preferred license). Replace this section if different.


