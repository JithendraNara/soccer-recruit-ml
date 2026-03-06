# Soccer Player Similarity & Recruitment Modeling Platform

Machine learning platform for evaluating player similarity across leagues and predicting recruitment value.

## Features

- **Player Similarity Model**: KNN-based similarity using cosine distance
- **Value Prediction**: Gradient Boosting regression for player valuation
- **REST API**: FastAPI endpoints for inference
- **Dashboard**: Streamlit visualization
- **MLflow Tracking**: Experiment tracking and model registry

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Load Sample Data

```bash
python scripts/load_data.py
```

### 3. Start API Server

```bash
uvicorn src.api.main:app --reload
```

### 4. Open Dashboard

```bash
streamlit run dashboards/app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/players` | GET | List players |
| `/api/v1/players/{id}` | GET | Get player details |
| `/api/v1/similarity/{id}/similar` | GET | Find similar players |
| `/api/v1/predict/value` | POST | Predict player value |
| `/health` | GET | Health check |

## Docker Deployment

```bash
docker-compose up --build
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Web framework
- **SQLAlchemy** - Database ORM
- **Scikit-learn** - ML models
- **MLflow** - Experiment tracking
- **Streamlit** - Dashboard
- **Plotly** - Visualizations

## Project Structure

```
soccer-recruit/
├── src/
│   ├── api/          # FastAPI app & routes
│   ├── ml/           # ML models & pipelines
│   ├── data/         # Database models & repos
│   └── utils/        # Config & logging
├── dashboards/       # Streamlit app
├── data/sample/     # Sample data
├── scripts/         # Utility scripts
└── tests/           # Test suite
```

## License

MIT
