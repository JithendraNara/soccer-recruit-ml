# Soccer Player Similarity & Recruitment Modeling Platform

## Project Overview

**Project Name:** SoccerRecruit ML
**Type:** Machine Learning Platform / Web API
**Core Functionality:** Evaluate player similarity across leagues using statistical modeling and feature embeddings, predict recruitment value, and provide REST API for model inference.
**Target Users:** Football scouts, recruitment analysts, sports data scientists

## Technology Stack

- **Language:** Python 3.11+
- **ML Tracking:** MLflow
- **Database:** SQLite (SQLAlchemy ORM)
- **Web Framework:** FastAPI
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Plotly, Streamlit (dashboards)

## Project Structure

```
soccer-recruit/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── routers/
│   │   │   ├── players.py       # Player endpoints
│   │   │   ├── similarity.py    # Similarity endpoints
│   │   │   └── prediction.py    # Prediction endpoints
│   │   └── schemas/
│   │       └── models.py       # Pydantic models
│   ├── ml/
│   │   ├── similarity/
│   │   │   ├── model.py         # Similarity model
│   │   │   └── trainer.py       # Model training
│   │   ├── prediction/
│   │   │   ├── model.py         # Value prediction model
│   │   │   └── trainer.py       # Model training
│   │   └── pipelines/
│   │       ├── ingestion.py     # Data ingestion
│   │       ├── cleaning.py      # Data cleaning
│   │       └── transformation.py # Feature engineering
│   ├── data/
│   │   ├── database.py         # SQLAlchemy setup
│   │   ├── models.py            # Database models
│   │   └── repositories.py     # Data access layer
│   └── utils/
│       ├── config.py            # Configuration
│       └── logger.py            # Logging setup
├── tests/
│   ├── unit/
│   ├── integration/
│   └── api/
├── data/
│   └── sample/                  # Sample data files
├── dashboards/
│   └── app.py                   # Streamlit dashboard
├── mlruns/                      # MLflow tracking
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
└── SPEC.md
```

## Functionality Specification

### 1. Data Pipeline

- **Ingestion:** Load CSV/JSON soccer data (player stats, match data)
- **Cleaning:** Handle missing values, outliers, normalize data
- **Transformation:** Create feature embeddings, compute derived metrics
- **Storage:** Persist to SQLite database

### 2. Player Similarity Model

- Feature extraction from player statistics
- Cosine similarity computation
- K-nearest neighbors for finding similar players
- MLflow tracking for experiments

### 3. Recruitment Value Prediction

- Regression model for player value estimation
- Time-series features from historical data
- Feature importance analysis

### 4. REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/players` | GET | List all players with filters |
| `/players/{id}` | GET | Get player details |
| `/players/{id}/similar` | GET | Find similar players |
| `/predict/value` | POST | Predict player value |
| `/health` | GET | API health check |

### 5. Dashboard

- Streamlit-based visualization
- Player comparison tools
- Similarity heatmaps
- Value prediction interface

## Data Models

### Player Features
- Personal: name, age, nationality, position
- Physical: height, weight
- Performance: goals, assists, appearances, minutes
- Advanced: pass accuracy, shots per game, tackles, interceptions
- Contract: value, wage, contract expiry

## Acceptance Criteria

1. FastAPI server starts without errors
2. Data pipeline ingests and processes sample data
3. Similarity API returns top-k similar players
4. Prediction API returns value estimates
5. MLflow tracks experiments
6. Dashboard renders player comparisons
7. All tests pass
8. Docker deployment works
