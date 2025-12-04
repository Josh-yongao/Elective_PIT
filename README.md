# Metro Manila Flood Risk Prediction

This repository contains code to train a Random Forest classifier to predict daily flood risk in Metro Manila using basic weather and station data, plus a Streamlit web app to run live predictions locally or after deployment.

Quick contents:
- `src/train.py` — training script that saves `models/rf_pipeline.joblib` and `models/metrics.json`.
- `src/model.py` — helper to load the saved pipeline and predict single samples.
- `streamlit_app.py` — Streamlit app for interactive predictions.
- `data/sample_data.csv` — small example dataset for testing and prototyping.
- `requirements.txt` — Python dependencies.

How to run locally

1. Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model (this will create `models/rf_pipeline.joblib`):

```bash
python src/train.py
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Deployment (Live Demo URL)

I cannot publish a live URL from this environment. To get a public demo URL you can use Streamlit Cloud (recommended):

1. Push this repository to GitHub.
2. On https://share.streamlit.io click "New app", connect your GitHub repo, and select the `streamlit_app.py` file as the entrypoint.
3. Streamlit Cloud will build and provide a public URL like `https://share.streamlit.io/<your-user>/<repo>/main`.

Example placeholder Live Demo URL (replace with your deployed URL):

`https://share.streamlit.io/your-username/Elective_PIT/main/streamlit_app.py`

Metric target guidance

Metric	Excellent Range	Notes
Accuracy	0.85–0.95	Data is imbalanced, so accuracy alone is not enough
Precision (Flood)	0.70–0.90	Higher precision means fewer false alarms
Recall (Flood)	0.75–0.95	MUST be high to avoid missing flood events
F1-Score (Flood)	0.75–0.92	Should be balanced with precision
ROC-AUC	0.85–0.95	Strong model performance

If you want, I can help deploy this repository to Streamlit Cloud and provide the real Live Demo URL — you'll need to grant deployment access or push the repo to your GitHub account and invite me (or paste the repo URL) so I can provide exact steps or finalize deployment.
# Elective_PIT