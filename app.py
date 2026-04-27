import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ── C:\Python314\python.exe -m streamlit run app.py ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0a0e1a; }
    .block-container { padding: 2rem 3rem; }

    h1 { font-family: 'Space Mono', monospace !important; color: #00ff88 !important; }
    h2, h3 { font-family: 'Space Mono', monospace !important; color: #e2e8f0 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00b4d8);
        color: #0a0e1a;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0,255,136,0.4);
    }

    .result-fraud {
        background: linear-gradient(135deg, #ff003322, #ff006644);
        border: 2px solid #ff0055;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-legit {
        background: linear-gradient(135deg, #00ff8822, #00b4d822);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-title { font-size: 2rem; font-weight: 700; font-family: 'Space Mono', monospace; }
    .result-sub   { font-size: 0.95rem; color: #94a3b8; margin-top: 0.4rem; }

    .info-box {
        background: #1e293b;
        border-left: 4px solid #00ff88;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    [data-testid="stNumberInput"] input {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    [data-testid="stTab"] { font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "fraud_xgb_model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Credit Card Fraud Detector")
st.markdown("**Real-time transaction analysis powered by XGBoost**")
st.divider()

if model is None:
    st.error("⚠️ Model file `fraud_xgb_model.pkl` not found. Place it in the same folder as `app.py`.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Manual Entry", "📂  Batch CSV Upload"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual entry
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Transaction Details")
    st.markdown(
        '<div class="info-box">Fill in the fields below (Time, V1–V16, V18–V28, Amount). '
        'V17 is auto-set to 0. V-features are PCA-transformed components.</div>',
        unsafe_allow_html=True,
    )

    with st.form("transaction_form"):
        col_t, col_a = st.columns(2)
        with col_t:
            time_val = st.number_input("⏱ Time (seconds elapsed)", value=0.0, format="%.4f")
        with col_a:
            amount_val = st.number_input("💲 Amount (USD)", value=0.0, min_value=0.0, format="%.4f")

        st.markdown("#### PCA Features (V1 – V16, V18 – V28)")
        v_vals = []
        cols = st.columns(4)
        col_idx = 0
        for i in range(1, 29):
            if i == 17:
                # V17 is skipped in UI but will be inserted as 0.0 later
                continue
            with cols[col_idx % 4]:
                v_vals.append(st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}"))
            col_idx += 1

        submitted = st.form_submit_button("🔎  Analyze Transaction")

    if submitted:
        # Build feature array: Time + V1-V28 (with V17=0) = 29 features
        features_list = [time_val]
        v_idx = 0
        for i in range(1, 29):
            if i == 17:
                features_list.append(0.0)  # V17 auto-set
            else:
                features_list.append(v_vals[v_idx])
                v_idx += 1
        # Amount is NOT passed to the model (model expects 29 features)
        features = np.array([features_list])

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        fraud_prob = proba[1] * 100
        legit_prob = proba[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <div class="result-title" style="color:#ff0055;">🚨 FRAUD DETECTED</div>
                <div class="result-sub">Fraud Probability: <strong style="color:#ff0055;">{fraud_prob:.2f}%</strong>
                &nbsp;|&nbsp; Legitimate Probability: {legit_prob:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                <div class="result-title" style="color:#00ff88;">✅ LEGITIMATE</div>
                <div class="result-sub">Legitimate Probability: <strong style="color:#00ff88;">{legit_prob:.2f}%</strong>
                &nbsp;|&nbsp; Fraud Probability: {fraud_prob:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Probability Breakdown")
        prob_df = pd.DataFrame({"Category": ["Legitimate", "Fraud"], "Probability (%)": [legit_prob, fraud_prob]})
        st.bar_chart(prob_df.set_index("Category"))


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV upload
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload CSV for Batch Prediction")
    st.markdown(
        '<div class="info-box">CSV must contain columns: '
        '<code>Time, V1, V2, ..., V28, Amount</code> (30 columns, in that order).</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded {len(df):,} transactions.** Preview:")
        st.dataframe(df.head(), width='stretch')

        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        missing = [c for c in expected_cols if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("🚀  Run Batch Prediction"):
                # Model expects 29 features: Time + V1-V28 (Amount excluded)
                model_cols = ["Time"] + [f"V{i}" for i in range(1, 29)]
                X = df[model_cols].values
                preds  = model.predict(X)
                probas = model.predict_proba(X)[:, 1]

                df["Prediction"]    = preds
                df["Fraud_Prob_%"]  = (probas * 100).round(2)
                df["Result"]        = df["Prediction"].map({0: "✅ Legitimate", 1: "🚨 Fraud"})

                total   = len(df)
                frauds  = int(preds.sum())
                legits  = total - frauds

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", f"{total:,}")
                c2.metric("🚨 Fraudulent",       f"{frauds:,}", delta=f"{frauds/total*100:.1f}%", delta_color="inverse")
                c3.metric("✅ Legitimate",        f"{legits:,}", delta=f"{legits/total*100:.1f}%")

                st.markdown("#### Results")
                result_display = df[["Time", "Amount", "Fraud_Prob_%", "Result"]]
                st.dataframe(result_display, width='stretch')

                # Export only result columns to prevent MemoryError
                output_cols = ["Time", "Amount", "Prediction", "Fraud_Prob_%", "Result"]
                csv_out = df[output_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download Full Results CSV",
                    csv_out,
                    "fraud_predictions.csv",
                    "text/csv",
                )