# Starter Plan for the S&P 500 Project

> A lightweight plan written like a student who just wants a working demo for class, not a perfect research project.

## 1. What I already have

- Historical dataset (`data.csv`) with prices, dividends, CPI, interest rates and PE10 values.
- Instructions describing the ideal end goal (many models + Flask dashboard).
- Checklist that is already marked as done, which means I can focus on packaging results and telling the story.

## 2. Minimum goals for a classroom presentation

1. Load the dataset, clean it and explain a few interesting trends.
2. Train a couple of easy models (Linear Regression for prices, Random Forest for up/down direction).
3. Show simple visuals (line chart of S&P 500, bar chart comparing model errors, confusion matrix).
4. Wrap the best parts in a short Flask page or, at least, a Jupyter notebook with Plotly graphs.
5. Prepare 10 slides that explain the motivation, data, models, results and lessons learned.

## 3. Step-by-step plan

| Step                            | Objective                                    | Concrete actions                                                                                                                                                                        | Deliverable                                                                           |
| ------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 0. Environment                  | Make sure Python runs smoothly               | Create a virtual env, install `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `plotly`, `flask`, `tensorflow` (only when I reach LSTM)                                                 | `requirements.txt` + instructions in README                                           |
| 1. Data loading & sanity checks | Prove I understand the dataset               | Write `src/data/load_data.py` with functions to read CSV, convert `Date` to datetime, set it as index, print shape/date range, check missing values                                     | Console output or notebook cells showing the checks                                   |
| 2. Quick EDA                    | Find a story to tell                         | In a notebook, plot S&P 500 nominal vs real price, CPI, PE10; compute rolling mean/volatility; save 3 charts for slides                                                                 | `notebooks/01_eda.ipynb` + exported PNGs                                              |
| 3. Feature prep                 | Build inputs for models without getting lost | Implement a simple pipeline that: fills missing data with forward fill, creates `return_1m`, `return_3m`, `return_12m`, and a binary column `direction` (`+1` if next month return > 0) | `data/processed/processed_data.csv`                                                   |
| 4. Baseline regression          | Predict next-month price level               | Train/train split by time (e.g., train until 2015, test after). Fit Linear Regression and optionally Ridge. Capture MAE/MSE.                                                            | `notebooks/02_regression_baseline.ipynb` or `src/models/regression.py` + metric table |
| 5. Baseline classification      | Predict up/down direction                    | Train Random Forest (default params) on same features. Produce accuracy, precision, recall, F1, confusion matrix plot.                                                                  | `notebooks/03_classification.ipynb` or script + saved model `rf.pkl`                  |
| 6. (Bonus) LSTM lite            | Have something “AI-like” to mention          | Reuse processed data, scale it, build sequences of 12 months, train a tiny LSTM (1 layer, 32 units, 10 epochs). Compare RMSE with Linear Regression.                                    | `notebooks/04_lstm.ipynb` + `lstm_model.h5`                                           |
| 7. Mini dashboard or demo       | Show results live                            | Option A: Flask app with 2 pages (home + predictions). Option B: Streamlit notebook-style app. Keep it simple: load saved metrics and show charts.                                      | `src/api/app.py` (Flask) or `app.py` (Streamlit)                                      |
| 8. Presentation prep            | Be ready for the classroom                   | Export best charts, copy metric tables, outline story: Problem → Data → Methods → Results → Lessons → Future work.                                                                      | `docs/slides.pptx` or `docs/slides.pdf`                                               |

## 4. Tips to keep it manageable

- **Reuse notebooks**: Use Jupyter/VSCode notebooks for EDA and models; later convert key functions into `.py` files if needed.
- **Track assumptions**: Write down every simplification (e.g., “I only tuned Random Forest trees between 50 and 150”). Professors like honesty.
- **Save intermediate data/models** so rerunning the demo is fast.
- **Version control**: Commit after each big milestone (EDA done, models trained, dashboard running).

## 5. Presentation cheat sheet

1. Slide 1 – Title & motivation (“Can we guess next month’s S&P 500 move?”).
2. Slide 2 – Dataset overview (timeline, key columns).
3. Slide 3 – Interesting trend (e.g., PE10 spikes before downturns).
4. Slide 4 – Feature engineering diagram (lags + returns).
5. Slide 5 – Regression model results (table of MAE/MSE).
6. Slide 6 – Classification model results + confusion matrix.
7. Slide 7 – LSTM screenshot or training curve (even if not perfect).
8. Slide 8 – Dashboard screenshot or notebook cells with Plotly charts.
9. Slide 9 – Limitations (data frequency, little hyperparameter tuning).
10. Slide 10 – Next steps (better tuning, more macro variables, deploy online).

## 6. Next moves for me

- [ ] Create virtual environment and install packages.
- [ ] Build the `load_data.py` helper and run first notebook.
- [ ] Finish EDA notebook and save 2–3 charts.
- [ ] Implement simple feature pipeline + baseline models.
- [ ] Decide whether the demo will be Flask or just a notebook + slides, then start assembling the presentation.

If I follow this plan, I’ll have a working prototype that I can explain confidently without needing every fancy technique from the original instructions.

## 7. Progress log

- **2025-11-16** – Organized the repository structure (src/, data/, notebooks/, docs/), moved the raw CSV to `data/raw/sp500.csv`, added a `requirements.txt` + `.gitignore`, and created `src/data/load_data.py`. Running `python -m src.data.load_data` now prints the dataset overview for quick sanity checks.
- **2025-11-23** – Completed Sprint 1 US-03: added the preprocessing pipeline (`src/data/pipeline.py`) plus demo script `scripts/Sprint01US03_Preprocessing_FeatureEngineering.py`. The script engineers lags/returns, creates regression & classification targets, scales features using a temporal split, and writes `data/processed/processed_data.csv`.
- **2025-12-16** – Completed Sprint 2 US-04: added `scripts/Sprint02US04_BaselineLinearModels.py` to train Linear, Ridge and Lasso on `processed_data.csv` with MAE/MSE/RMSE output.
- **2025-12-16** – Completed Sprint 2 US-05: added `scripts/Sprint02US05_Classification_RF_GBM.py` to train Random Forest and Gradient Boosting on `target_direction`, print Accuracy/Precision/Recall/F1 plus confusion matrices, and save `models/rf.pkl` and `models/gbm.pkl`.
