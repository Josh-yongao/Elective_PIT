import streamlit as st
import os
import json
import pandas as pd
from src.model import load_model, predict_single

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'metrics.json')

st.set_page_config(page_title='Metro Manila Flood Risk Predictor', layout='centered')

st.title('Metro Manila Daily Flood Risk Predictor')
st.write('Enter daily weather and station information to get a flood risk prediction.')

# --- Simple CSS to improve visuals ---
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; color: #e6eef6 }
    .big-title {font-size:40px; font-weight:700; color:#ffffff;}
    .subtitle {color:#cbd5e1; margin-bottom:20px}
    .card {background-color:#0b1220; padding:18px; border-radius:10px;}
    .metric-label {color:#94a3b8}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="subtitle">Forecast daily flood risk for Metro Manila stations — model trained on historical PAGASA/MMDA data.</div>', unsafe_allow_html=True)

with st.sidebar.expander('Model target ranges'):
    st.markdown('''
- **Accuracy**: 0.85–0.95  
- **Precision (Flood)**: 0.70–0.90  
- **Recall (Flood)**: 0.75–0.95  
- **F1-Score (Flood)**: 0.75–0.92  
- **ROC-AUC**: 0.85–0.95  
''')


# Load model and infer expected features and threshold
model = None
best_threshold = 0.5
metrics = None
try:
    model = load_model()
    preprocessor = model.named_steps.get('preprocessor') if hasattr(model, 'named_steps') else None
    numeric_features = []
    categorical_features = []
    rainfall_bin_options = ['low', 'moderate', 'high', 'extreme']
    if preprocessor is not None:
        for name, transformer, cols in preprocessor.transformers:
            if cols == 'drop' or cols is None:
                continue
            if isinstance(cols, (list, tuple)):
                if name == 'num':
                    numeric_features = list(cols)
                elif name == 'cat':
                    categorical_features = list(cols)
    # Load threshold and metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        # Use test threshold if available, else validation
        best_threshold = metrics.get('test', {}).get('best_threshold', metrics.get('validation', {}).get('best_threshold', 0.5))
except Exception as e:
    st.error(f'Error loading model: {e}')

# Metrics panel
with st.container():
    if metrics:
        tm = metrics.get('test', {})
        vm = metrics.get('validation', {})
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Model Performance (Test)')
        cols = st.columns(5)
        cols[0].metric('Accuracy', f"{tm.get('accuracy',0):.3f}")
        cols[1].metric('Precision (Flood)', f"{tm.get('precision',0):.3f}")
        cols[2].metric('Recall (Flood)', f"{tm.get('recall',0):.3f}")
        cols[3].metric('F1 (Flood)', f"{tm.get('f1',0):.3f}")
        cols[4].metric('ROC-AUC', f"{tm.get('roc_auc',0):.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info('No saved model metrics found. Run training first.')



# Build inputs dynamically based on model features (single section, unique keys)
inputs = {}
st.subheader('Input Features')
cols = st.columns(2)
if preprocessor is None:
    # Fallback inputs
    inputs['rainfall'] = cols[0].number_input('Rainfall (mm)', value=10.0, step=0.1, key='rainfall_input')
    inputs['elevation'] = cols[1].number_input('Elevation (m)', value=10.0, step=0.1, key='elevation_input')
    inputs['location'] = st.selectbox('Location', options=['Quezon City','Marikina','Manila','Pasig'], key='location_input')
    inputs['rainfall_bin'] = 'moderate'
else:
    # Numeric features
    num_cols = [c for c in numeric_features]
    for i, feat in enumerate(num_cols):
        col = cols[i % 2]
        default = 0.0
        try:
            default = float(preprocessor.named_transformers_['num'].named_steps.get('imputer').statistics_[num_cols.index(feat)])
        except Exception:
            default = 0.0
        inputs[feat] = col.number_input(f'{feat}', value=default, step=0.1, key=f'num_{feat}')

    # Categorical features
    for feat in categorical_features:
        opts = ['Unknown']
        try:
            cat_pipeline = preprocessor.named_transformers_.get('cat')
            onehot = cat_pipeline.named_steps.get('onehot') if cat_pipeline is not None else None
            if onehot is not None:
                idx = categorical_features.index(feat)
                opts = list(onehot.categories_[idx]) if len(onehot.categories_) > idx else ['Unknown']
        except Exception:
            pass
        inputs[feat] = st.selectbox(feat, options=opts, key=f'cat_{feat}')

    # Ensure rainfall_bin exists (compute from rainfall if not present)
    if 'rainfall_bin' not in inputs:
        def get_rainfall_bin(val):
            if val <= 5:
                return 'low'
            elif val <= 15:
                return 'moderate'
            elif val <= 30:
                return 'high'
            else:
                return 'extreme'
        # choose a rainfall feature name if present
        rain_feat = 'rainfall' if 'rainfall' in inputs else (numeric_features[0] if numeric_features else None)
        rain_val = inputs[rain_feat] if rain_feat else 0.0
        inputs['rainfall_bin'] = get_rainfall_bin(rain_val)

# Only one button, outside all input logic
btn_col = st.columns([1,2,1])
with btn_col[1]:
    predict = st.button('Predict Flood Risk', key='predict_flood_button')

if predict:
    try:
        if model is None:
            st.error('Model not loaded. Train the model first.')
        else:
            # Prepare input DataFrame in the same way as training
            df_in = pd.DataFrame([inputs])
            proba = model.predict_proba(df_in)[0, 1]
            pred = int(proba >= best_threshold)

            # Result card
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader('Prediction Result')
                prob_pct = proba * 100
                st.metric('Flood Probability', f"{prob_pct:.1f}%")
                st.markdown(f"**Threshold used:** {best_threshold:.2f}")
                if pred == 1:
                    st.error('High flood risk — notify authorities and prepare evacuation if necessary.')
                else:
                    st.success('Low flood risk for the day.')
                st.markdown('</div>', unsafe_allow_html=True)

            if metrics:
                st.subheader('Model Metrics (Test set)')
                tm = metrics.get('test', {})
                st.write(tm)
            else:
                st.info('No saved model metrics found. Run training first.')
    except Exception as e:
        st.error(f'Error during prediction: {e}')

# Removed duplicate button block - predictions handled above when 'predict' is True

st.markdown('---')
st.markdown('''
To deploy this app as a Live Demo, push this repository to GitHub and deploy with Streamlit Cloud: https://share.streamlit.io
Follow the README instructions to obtain the live URL.
''')
