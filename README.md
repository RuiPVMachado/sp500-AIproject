# S&P 500 Classroom Project

A humble university project that predicts the S&P 500 with a couple of classic models, a tiny LSTM experiment, and a small Flask/Plotly demo. The goal is to have a working prototype we can explain during a presentation, not a production-ready quant platform.

## Repository layout

```
sp500-AIproject/
├── checklist.md              # Original checklist provided by the professor
├── instructions.md           # Detailed instructions / requirements
├── project_plan.md           # Student-friendly action plan
├── data/
│   ├── raw/                  # Original CSV files (sp500.csv already included)
│   └── processed/            # Outputs from the preprocessing pipeline
├── docs/                     # Slides, EDA summaries, anomaly reports, etc.
├── notebooks/                # Exploratory notebooks (EDA, models, LSTM)
├── src/
│   ├── data/                 # Data loading & preprocessing helpers
│   ├── models/               # Training scripts for regression/classification
│   └── api/                  # Flask app and utility modules
├── models/                   # Serialized models (rf.pkl, gbm.pkl, lstm_model.h5)
├── requirements.txt
└── README.md
```

## Quickstart

1. **Create a virtual environment (recommended)**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Load and inspect the dataset**
   ```powershell
   python -m src.data.load_data
   ```
   This will print the date range, number of rows, column dtypes and missing-value counts.

## Next milestones

- Finish the exploratory notebook (`notebooks/01_eda.ipynb`) and export 2–3 charts for the slides.
- Expand the preprocessing pipeline (lags, returns, direction flag) and save `data/processed/processed_data.csv`.
- Train Linear/Ridge/Lasso for regression and RandomForest/GradientBoosting for classification.
- Capture metrics + confusion matrix images for the presentation.
- Build the lightweight Flask dashboard or a Streamlit alternative if time is short.

## Notes for teammates

- Keep code modular and simple; pretend you’re explaining it to classmates.
- Save intermediate outputs in `data/processed/` and trained models inside `models/` so demos load quickly.
- Whenever you try a new idea, jot it down in `project_plan.md` or a short notebook cell so we remember the story for the presentation.
