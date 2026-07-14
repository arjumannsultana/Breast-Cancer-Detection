import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

# ── CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&display=swap');

.stApp {
    background: #f0f4f8;
    color: #1a202c;
}

section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4, p, label, div, span, button {
    font-family: 'DM Sans', sans-serif;
}

.result-malignant {
    background: rgba(248,113,113,0.08);
    border: 2px solid #f87171;
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
}
.result-benign {
    background: rgba(52,211,153,0.08);
    border: 2px solid #34d399;
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
}
.result-empty {
    background: #e2e8f0;
    border: 2px dashed #cbd5e0;
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
    color: #a0aec0;
}

.vote-bar-wrap {
    background: #cbd5e0;
    border-radius: 100px;
    height: 10px;
    margin: 6px 0 14px 0;
    overflow: hidden;
}
.vote-bar-fill-m {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #f87171, #e05ca0);
}
.vote-bar-fill-b {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #34d399, #2dd4bf);
}

.feature-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px 20px 4px 20px;
    margin-bottom: 12px;
}
.feature-section h4 {
    color: #718096;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
    font-weight: 700;
}

.stSlider label {
    font-weight: 700 !important;
    color: #2d3748 !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #e05ca0, #7b5ea7) !important;
}

.hero-banner {
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 60%, #4a235a 100%);
    border-radius: 20px;
    padding: 48px 40px 36px 40px;
    text-align: center;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: rgba(224,92,160,0.15);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50px; left: -30px;
    width: 220px; height: 220px;
    background: rgba(123,94,167,0.12);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 48px;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
    position: relative;
    z-index: 1;
}

/* Primary button — active tab + analyze tumor */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #e05ca0, #7b5ea7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 100px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 8px 28px !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0px !important;
}
div[data-testid="stButton"] > button[kind="primary"] p {
    color: #ffffff !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.88 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover p {
    color: #ffffff !important;
}

/* Secondary button — inactive tab */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #ffffff !important;
    color: #4a5568 !important;
    border: 1.5px solid #cbd5e0 !important;
    border-radius: 100px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 8px 28px !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #e05ca0 !important;
    color: #e05ca0 !important;
}

/* Analyze Tumor button — override radius to be rectangular */
div[data-testid="stButton"]:has(button[key="analyze"]) > button {
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px !important;
    letter-spacing: 0.5px !important;
    color: #ffffff !important;
}

.disclaimer {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 12px;
    color: #92400e;
    margin-top: 20px;
    font-weight: 600;
}

.locked-box {
    background: #e2e8f0;
    border: 2px dashed #cbd5e0;
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
}

.report-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    border-top: 4px solid;
}
.report-card.pink   { border-top-color: #e05ca0; }
.report-card.teal   { border-top-color: #2dd4bf; }
.report-card.purple { border-top-color: #7b5ea7; }
.report-card.amber  { border-top-color: #f59e0b; }
.report-card h3 {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #1a202c !important;
    margin: 0 0 8px 0 !important;
}
.report-card p {
    font-size: 13px;
    color: #4a5568;
    margin: 0 0 10px 0;
    line-height: 1.6;
}
.report-card ul {
    margin: 0;
    padding-left: 18px;
    color: #4a5568;
    font-size: 13px;
    line-height: 1.8;
}

h2, h3 { font-weight: 700 !important; color: #2d3748 !important; }
p { color: #4a5568; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler ─────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

FEATURE_NAMES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean',
    'concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se',
    'smoothness_se','compactness_se','concavity_se',
    'concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst',
    'smoothness_worst','compactness_worst','concavity_worst',
    'concave_points_worst','symmetry_worst','fractal_dimension_worst'
]

FEATURE_DEFAULTS = {
    'radius_mean': 14.13, 'texture_mean': 19.29, 'perimeter_mean': 91.97,
    'area_mean': 654.89, 'smoothness_mean': 0.0964, 'compactness_mean': 0.1043,
    'concavity_mean': 0.0888, 'concave_points_mean': 0.0489,
    'symmetry_mean': 0.1812, 'fractal_dimension_mean': 0.0628,
    'radius_se': 0.4052, 'texture_se': 1.2169, 'perimeter_se': 2.866,
    'area_se': 40.34, 'smoothness_se': 0.00704, 'compactness_se': 0.02548,
    'concavity_se': 0.03189, 'concave_points_se': 0.01180,
    'symmetry_se': 0.02054, 'fractal_dimension_se': 0.003795,
    'radius_worst': 16.27, 'texture_worst': 25.68, 'perimeter_worst': 107.26,
    'area_worst': 880.58, 'smoothness_worst': 0.1324, 'compactness_worst': 0.2543,
    'concavity_worst': 0.2722, 'concave_points_worst': 0.1146,
    'symmetry_worst': 0.2901, 'fractal_dimension_worst': 0.08395
}

BENIGN_AVG = {
    'radius_mean': 12.15, 'texture_mean': 17.91, 'perimeter_mean': 78.07,
    'area_mean': 462.79, 'smoothness_mean': 0.0925, 'compactness_mean': 0.0801,
    'concavity_mean': 0.0461, 'concave_points_mean': 0.0257,
    'symmetry_mean': 0.1748, 'fractal_dimension_mean': 0.0629
}
MALIGNANT_AVG = {
    'radius_mean': 17.46, 'texture_mean': 21.60, 'perimeter_mean': 115.37,
    'area_mean': 978.38, 'smoothness_mean': 0.1028, 'compactness_mean': 0.1447,
    'concavity_mean': 0.1600, 'concave_points_mean': 0.0879,
    'symmetry_mean': 0.1928, 'fractal_dimension_mean': 0.0627
}

# ── Session state ────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'detection'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = None
if 'votes' not in st.session_state:
    st.session_state.votes = None
if 'last_values' not in st.session_state:
    st.session_state.last_values = None


# ── Hero Banner ──────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Breast Cancer Detection</div>
</div>
""", unsafe_allow_html=True)

# ── Tab Navigation ───────────────────────────────────
col_gap1, col_d, col_dd, col_gap2 = st.columns([2.5, 1, 1, 2.5])
with col_d:
    if st.button(
        "🔍 Detection",
        key="nav_detection",
        type="primary" if st.session_state.page == 'detection' else "secondary"
    ):
        st.session_state.page = 'detection'
        st.rerun()
with col_dd:
    if st.button(
        "🧬 Deep Dive",
        key="nav_deepdive",
        type="primary" if st.session_state.page == 'deepdive' else "secondary"
    ):
        st.session_state.page = 'deepdive'
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 1 — DETECTION
# ══════════════════════════════════════════════════════
if st.session_state.page == 'detection':

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("<p style='font-size:13px;color:#718096;font-weight:600;margin-bottom:16px;'>Adjust each slider based on the pathology report values.</p>", unsafe_allow_html=True)

        st.markdown('<div class="feature-section"><h4>📐 Size & Shape</h4>', unsafe_allow_html=True)
        radius = st.slider(
            "Tumor Radius (mm)",
            min_value=6.98, max_value=28.11, value=14.13, step=0.1,
            help="Mean radius of the tumor cell nuclei")
        perimeter = st.slider(
            "Tumor Perimeter (mm)",
            min_value=43.79, max_value=188.50, value=91.97, step=0.5,
            help="Outer boundary length of the tumor")
        area = st.slider(
            "Tumor Area (mm²)",
            min_value=143.50, max_value=2501.00, value=654.89, step=5.0,
            help="Total size of the tumor")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-section"><h4>🔬 Texture & Smoothness</h4>', unsafe_allow_html=True)
        texture = st.slider(
            "Texture (higher = rougher surface)",
            min_value=9.71, max_value=39.28, value=19.29, step=0.1,
            help="Standard deviation of gray-scale values in the cell image")
        smoothness = st.slider(
            "Smoothness (0 = very smooth)",
            min_value=0.05, max_value=0.16, value=0.096, step=0.001,
            help="Local variation in radius lengths")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-section"><h4>🔷 Shape Irregularity</h4>', unsafe_allow_html=True)
        compactness = st.slider(
            "Compactness (higher = more irregular)",
            min_value=0.02, max_value=0.35, value=0.104, step=0.001,
            help="Perimeter² / Area − 1.0")
        concavity = st.slider(
            "Concavity (higher = deeper indentations)",
            min_value=0.0, max_value=0.43, value=0.089, step=0.001,
            help="Severity of concave portions of the tumor contour")
        concave_points = st.slider(
            "Concave Points (higher = more indentations)",
            min_value=0.0, max_value=0.20, value=0.049, step=0.001,
            help="Number of concave portions of the contour")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-section"><h4>⚖️ Symmetry & Complexity</h4>', unsafe_allow_html=True)
        symmetry = st.slider(
            "Symmetry (lower = more symmetric)",
            min_value=0.106, max_value=0.304, value=0.181, step=0.001,
            help="How evenly shaped the tumor is")
        fractal = st.slider(
            "Border Complexity (higher = more jagged)",
            min_value=0.05, max_value=0.097, value=0.063, step=0.001,
            help="Coastline approximation of the tumor border")
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("Analyze Tumor", type="primary", key="analyze")

    with right:
        st.markdown("### 📊 Result")

        # ── Show empty state only if no prediction ever made
        if st.session_state.prediction is None and not predict_btn:
            st.markdown("""
            <div class="result-empty">
                <div style="font-size:48px; margin-bottom:16px;">🔬</div>
                <p style="color:#a0aec0; font-size:15px;">
                    Fill in the measurements on the left<br>and click <b>Analyze Tumor</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Run prediction on button click
        if predict_btn:
            user_inputs = {
                'radius_mean': radius, 'texture_mean': texture,
                'perimeter_mean': perimeter, 'area_mean': area,
                'smoothness_mean': smoothness, 'compactness_mean': compactness,
                'concavity_mean': concavity, 'concave_points_mean': concave_points,
                'symmetry_mean': symmetry, 'fractal_dimension_mean': fractal,
            }
            full_input  = [user_inputs.get(f, FEATURE_DEFAULTS[f]) for f in FEATURE_NAMES]
            arr         = np.array(full_input).reshape(1, -1)
            arr_scaled  = scaler.transform(arr)
            prediction  = model.predict(arr_scaled)[0]

            distances, indices = model.kneighbors(arr_scaled)
            neighbor_labels = model._y[indices[0]]
            malignant_votes = int(np.sum(neighbor_labels == 1))
            benign_votes    = int(np.sum(neighbor_labels == 0))
            k               = len(neighbor_labels)

            st.session_state.prediction  = prediction
            st.session_state.user_inputs = user_inputs
            st.session_state.votes       = (malignant_votes, benign_votes, k)
            st.session_state.last_values = {
                'radius': radius, 'texture': texture,
                'perimeter': perimeter, 'area': area,
                'smoothness': smoothness, 'compactness': compactness,
                'concavity': concavity, 'concave_points': concave_points,
                'symmetry': symmetry, 'fractal': fractal
            }

        # ── Display result if prediction exists (new or returning)
        if st.session_state.prediction is not None:
            prediction      = st.session_state.prediction
            malignant_votes, benign_votes, k = st.session_state.votes
            mal_pct = int(malignant_votes / k * 100)
            ben_pct = int(benign_votes / k * 100)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-malignant">
                    <div style="font-size:52px; margin-bottom:8px;">🔴</div>
                    <h1 style="color:#e53e3e !important; font-size:38px; margin:0;
                        font-family:'DM Serif Display',serif;">MALIGNANT</h1>
                    <p style="color:#c53030; font-size:15px; margin-top:10px; font-weight:600;">
                        The tumor shows characteristics consistent with a malignant mass.<br>
                        <b>Immediate medical consultation is strongly advised.</b>
                    </p>
                    <div style="margin-top:24px; text-align:left;">
                        <div style="display:flex; justify-content:space-between; font-size:13px;
                            color:#718096; margin-bottom:4px; font-weight:600;">
                            <span>🔴 Malignant votes</span>
                            <span><b style="color:#e53e3e">{malignant_votes}/{k}</b></span>
                        </div>
                        <div class="vote-bar-wrap">
                            <div class="vote-bar-fill-m" style="width:{mal_pct}%"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px;
                            color:#718096; margin-bottom:4px; font-weight:600;">
                            <span>🟢 Benign votes</span>
                            <span><b style="color:#38a169">{benign_votes}/{k}</b></span>
                        </div>
                        <div class="vote-bar-wrap">
                            <div class="vote-bar-fill-b" style="width:{ben_pct}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-benign">
                    <div style="font-size:52px; margin-bottom:8px;">🟢</div>
                    <h1 style="color:#38a169 !important; font-size:38px; margin:0;
                        font-family:'DM Serif Display',serif;">BENIGN</h1>
                    <p style="color:#276749; font-size:15px; margin-top:10px; font-weight:600;">
                        The tumor shows characteristics consistent with a benign mass.<br>
                        <b>Continue regular monitoring and check-ups.</b>
                    </p>
                    <div style="margin-top:24px; text-align:left;">
                        <div style="display:flex; justify-content:space-between; font-size:13px;
                            color:#718096; margin-bottom:4px; font-weight:600;">
                            <span>🟢 Benign votes</span>
                            <span><b style="color:#38a169">{benign_votes}/{k}</b></span>
                        </div>
                        <div class="vote-bar-wrap">
                            <div class="vote-bar-fill-b" style="width:{ben_pct}%"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px;
                            color:#718096; margin-bottom:4px; font-weight:600;">
                            <span>🔴 Malignant votes</span>
                            <span><b style="color:#e53e3e">{malignant_votes}/{k}</b></span>
                        </div>
                        <div class="vote-bar-wrap">
                            <div class="vote-bar-fill-m" style="width:{mal_pct}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Values table — use saved values when returning
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📝 Values You Entered")
            v = st.session_state.last_values
            summary = pd.DataFrame({
                'Measurement': [
                    'Radius (mm)', 'Texture', 'Perimeter (mm)', 'Area (mm²)',
                    'Smoothness', 'Compactness', 'Concavity',
                    'Concave Points', 'Symmetry', 'Border Complexity'
                ],
                'Value': [
                    v['radius'], v['texture'], v['perimeter'], v['area'],
                    v['smoothness'], v['compactness'], v['concavity'],
                    v['concave_points'], v['symmetry'], v['fractal']
                ]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Disclaimer:</b> This tool is built for academic purposes only.
            It is not a substitute for professional medical diagnosis.
            Always consult a qualified medical professional.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — DEEP DIVE
# ══════════════════════════════════════════════════════
elif st.session_state.page == 'deepdive':

    st.markdown("### 📡 Your Tumor Profile")
    st.markdown("<p style='color:#718096;font-size:13px;margin-top:-10px;margin-bottom:24px;'>How your measurements compare to the typical malignant and benign profiles in the dataset.</p>", unsafe_allow_html=True)

    if st.session_state.prediction is None:
        st.markdown("""
        <div class="locked-box">
            <div style="font-size:48px; margin-bottom:16px;">🔒</div>
            <p style="color:#718096; font-size:16px; font-weight:600;">
                No prediction made yet.<br>
                Go to <b>🔍 Detection</b> and click <b>Analyze Tumor</b> first.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        user_inputs     = st.session_state.user_inputs
        prediction      = st.session_state.prediction
        malignant_votes, benign_votes, k = st.session_state.votes

        feature_keys   = list(BENIGN_AVG.keys())
        features_short = ['Radius', 'Texture', 'Perimeter', 'Area',
                          'Smoothness', 'Compactness', 'Concavity',
                          'Concave Pts', 'Symmetry', 'Border\nComplexity']

        def normalize(values, keys):
            norm = []
            for i, k_name in enumerate(keys):
                mn = min(BENIGN_AVG[k_name], MALIGNANT_AVG[k_name]) * 0.8
                mx = max(BENIGN_AVG[k_name], MALIGNANT_AVG[k_name]) * 1.2
                norm.append((values[i] - mn) / (mx - mn + 1e-9))
            return norm

        user_vals   = [user_inputs[k] for k in feature_keys]
        benign_vals = [BENIGN_AVG[k]  for k in feature_keys]
        mal_vals    = [MALIGNANT_AVG[k] for k in feature_keys]

        user_norm   = normalize(user_vals,   feature_keys)
        benign_norm = normalize(benign_vals, feature_keys)
        mal_norm    = normalize(mal_vals,    feature_keys)

        N      = len(features_short)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        user_norm   += user_norm[:1]
        benign_norm += benign_norm[:1]
        mal_norm    += mal_norm[:1]

        col_r, col_s = st.columns([1.2, 0.8], gap="large")

        with col_r:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#f0f4f8')
            ax.set_facecolor('#f0f4f8')

            ax.plot(angles, mal_norm,    color='#f87171', linewidth=2)
            ax.fill(angles, mal_norm,    color='#f87171', alpha=0.1)
            ax.plot(angles, benign_norm, color='#34d399', linewidth=2)
            ax.fill(angles, benign_norm, color='#34d399', alpha=0.1)
            ax.plot(angles, user_norm,   color='#7b5ea7', linewidth=2.5)
            ax.fill(angles, user_norm,   color='#7b5ea7', alpha=0.15)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features_short, size=9, color='#4a5568', fontweight='bold')
            ax.set_yticklabels([])
            ax.grid(color='#cbd5e0', linewidth=0.8)
            ax.spines['polar'].set_color('#cbd5e0')

            mal_patch    = mpatches.Patch(color='#f87171', label='Avg Malignant')
            benign_patch = mpatches.Patch(color='#34d399', label='Avg Benign')
            user_patch   = mpatches.Patch(color='#7b5ea7', label='Your Values')
            ax.legend(handles=[user_patch, mal_patch, benign_patch],
                      loc='upper right', bbox_to_anchor=(1.35, 1.15),
                      fontsize=10, framealpha=0, labelcolor='#2d3748')
            st.pyplot(fig)

        with col_s:
            result_label = "MALIGNANT" if prediction == 1 else "BENIGN"
            result_color = "#e53e3e"   if prediction == 1 else "#38a169"

            closer_to_mal = sum(
                1 for k_name in feature_keys
                if abs(user_inputs[k_name] - MALIGNANT_AVG[k_name])
                   < abs(user_inputs[k_name] - BENIGN_AVG[k_name])
            )
            closer_to_ben = 10 - closer_to_mal

            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
                padding:24px; margin-bottom:16px;">
                <p style="font-size:13px; color:#718096; font-weight:600; margin-bottom:4px;">
                    PREDICTION
                </p>
                <h2 style="color:{result_color} !important; font-size:28px; margin:0 0 16px 0;
                    font-family:'DM Serif Display',serif;">{result_label}</h2>
                <p style="font-size:14px; color:#4a5568; line-height:1.7; margin:0;">
                    Your profile sits closer to the
                    <b style="color:#e53e3e">malignant average</b> across
                    <b>{closer_to_mal} of 10 features</b> and closer to the
                    <b style="color:#38a169">benign average</b> across
                    <b>{closer_to_ben} of 10 features</b>.
                </p>
            </div>
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
                padding:24px;">
                <p style="font-size:13px; color:#718096; font-weight:600; margin-bottom:12px;">
                    NEIGHBOR VOTES
                </p>
                <div style="display:flex; gap:16px;">
                    <div style="text-align:center;">
                        <div style="font-size:28px; font-weight:700; color:#e53e3e;">
                            {malignant_votes}
                        </div>
                        <div style="font-size:12px; color:#718096; font-weight:600;">Malignant</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:28px; font-weight:700; color:#38a169;">
                            {benign_votes}
                        </div>
                        <div style="font-size:12px; color:#718096; font-weight:600;">Benign</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:28px; font-weight:700; color:#7b5ea7;">{k}</div>
                        <div style="font-size:12px; color:#718096; font-weight:600;">Total (k)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Understanding Your Report ────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Understanding Your Report")
    st.markdown("<p style='color:#718096;font-size:13px;margin-top:-10px;margin-bottom:24px;'>What each measurement on a pathology report actually means.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="report-card pink">
            <h3>📐 Size & Shape</h3>
            <p>These measurements describe the physical dimensions of the tumor cell nuclei
            seen under a microscope.</p>
            <ul>
                <li><b>Radius</b> — average distance from the center to the edge of the nucleus</li>
                <li><b>Perimeter</b> — the boundary length around the nucleus</li>
                <li><b>Area</b> — total size of the nucleus</li>
            </ul>
            <br>
            <p style="color:#e05ca0; font-size:12px; font-weight:700;">
                ⚠️ Malignant tumors are typically larger — higher radius, perimeter and area.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="report-card purple">
            <h3>🔷 Shape Irregularity</h3>
            <p>These describe how irregular or distorted the shape of the tumor nucleus is.</p>
            <ul>
                <li><b>Compactness</b> — how round the shape is. A perfect circle = 0</li>
                <li><b>Concavity</b> — how deeply indented the edges are</li>
                <li><b>Concave Points</b> — how many indentations exist on the edge</li>
            </ul>
            <br>
            <p style="color:#7b5ea7; font-size:12px; font-weight:700;">
                ⚠️ Malignant cells tend to have highly irregular, indented shapes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="report-card teal">
            <h3>🔬 Texture & Smoothness</h3>
            <p>These describe the surface quality of the tumor cell as seen in a microscope image.</p>
            <ul>
                <li><b>Texture</b> — how much the gray values vary across the cell image.
                Higher = rougher, more uneven surface</li>
                <li><b>Smoothness</b> — how consistent the radius is around the cell.
                Lower = smoother, more regular</li>
            </ul>
            <br>
            <p style="color:#2dd4bf; font-size:12px; font-weight:700;">
                ⚠️ Malignant cells often appear rougher and less uniform in texture.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="report-card amber">
            <h3>⚖️ Symmetry & Complexity</h3>
            <p>These describe how structured and regular the overall cell shape is.</p>
            <ul>
                <li><b>Symmetry</b> — how evenly shaped the nucleus is on both sides.
                Benign cells tend to be more symmetric</li>
                <li><b>Border Complexity</b> — how jagged or irregular the outline is.
                A smooth circle has low complexity</li>
            </ul>
            <br>
            <p style="color:#f59e0b; font-size:12px; font-weight:700;">
                ⚠️ Malignant cells often show higher asymmetry and irregular borders.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
        padding:24px 28px; margin-top:4px;">
        <h3 style="margin-top:0; font-size:15px !important;">🔬 Where Do These Numbers Come From?</h3>
        <p style="font-size:13px; color:#4a5568; line-height:1.8; margin:0;">
            These measurements are extracted from digitized images of a
            <b>Fine Needle Aspirate (FNA)</b> — a procedure where a thin needle is used
            to collect a small sample of cells from the breast mass.
            The sample is placed on a slide, stained, and photographed under a microscope.
            Software then automatically computes these 10 measurements from the cell nuclei
            visible in the image. A pathologist reviews both the image and the measurements
            to make a diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────
st.markdown("""
<hr style="border:none; border-top:1px solid #e2e8f0; margin: 40px 0 20px 0;">
<p style="text-align:center; color:#a0aec0; font-size:13px; font-weight:600;">
    Built by <b style="color:#e05ca0;">Arjuman Sultana</b> · AI & Data Science · KNN Classifier
</p>
""", unsafe_allow_html=True)