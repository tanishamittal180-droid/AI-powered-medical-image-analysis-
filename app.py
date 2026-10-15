from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.inference import MODEL_ID, predict_xray

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="MedVision AI | Chest X-ray Demo", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --mv-ink:#15243b; --mv-muted:#607086; --mv-blue:#3267e8; }
.block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px;}
.hero {padding: 1.7rem 1.8rem; border-radius: 22px; background: linear-gradient(125deg,#14223b 0%,#244a85 60%,#3474e8 100%); color:white; margin-bottom:1.2rem;}
.hero h1 {font-size:2.2rem; margin:0 0 .4rem 0; color:white;}
.hero p {font-size:1.02rem; opacity:.92; margin:0;}
.metric-card {border:1px solid #e2e8f0; border-radius:16px; padding:1rem 1.1rem; background:rgba(255,255,255,.7); min-height:100px;}
.metric-label {font-size:.84rem; color:#607086;}
.metric-value {font-size:1.5rem; font-weight:750; color:#15243b; margin-top:.35rem;}
.small-note {color:#607086;font-size:.88rem;}
</style>
<div class="hero"><h1>🩻 MedVision AI</h1><p>Chest X-ray classification demo · pretrained deep-learning model · image analysis and exportable results</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page", ["X-ray Analyzer", "About the Model", "Project Guide"], label_visibility="collapsed")
    st.divider()
    st.caption("MODEL SOURCE")
    st.markdown(f"[{MODEL_ID}](https://huggingface.co/{MODEL_ID})")
    st.caption("First run requires internet to download model files. Later runs may use the local cache.")

if page == "X-ray Analyzer":
    st.subheader("Analyze a chest X-ray")
    st.warning("Educational demonstration only. This model distinguishes NORMAL from PNEUMONIA in chest X-rays; it does not detect every disease and must not be used for diagnosis or treatment decisions.")
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg", "webp", "bmp"], help="Use a frontal chest X-ray image. Do not upload identifiable patient information.")
        if uploaded:
            try:
                raw = Image.open(io.BytesIO(uploaded.getvalue()))
                image = ImageOps.exif_transpose(raw).convert("RGB")
                st.image(image, caption=f"Uploaded image · {image.width} × {image.height}px", use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError):
                image = None
                st.error("This file could not be opened as an image. Please upload a valid PNG or JPEG.")
        else:
            image = None
            st.info("Upload an image to enable analysis.")
        run = st.button("Run X-ray analysis", type="primary", use_container_width=True, disabled=image is None)

    with right:
        if run and image is not None:
            with st.spinner("Loading the pretrained model and analyzing the image. The first run can take several minutes..."):
                try:
                    result = predict_xray(image)
                    st.session_state["last_result"] = result
                    st.session_state["last_filename"] = uploaded.name if uploaded else "uploaded_xray"
                except Exception as exc:
                    st.error("The model could not run. Check the internet connection, installed packages, and terminal error, then retry.")
                    with st.expander("Technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")
        result = st.session_state.get("last_result")
        if result:
            label = result["label"]
            score = result["score"]
            st.markdown("### Model result")
            if label == "PNEUMONIA":
                st.error("Model class: PNEUMONIA")
            elif label == "NORMAL":
                st.success("Model class: NORMAL")
            else:
                st.info(f"Model class: {label}")
            st.metric("Top class score", f"{score * 100:.1f}%")
            st.caption("This is the model's classification score, not a calibrated probability that a person has a disease.")
            scores = result["scores"]
            score_df = pd.DataFrame({"Class": list(scores.keys()), "Score (%)": [round(v * 100, 2) for v in scores.values()]})
            st.markdown("#### Class scores")
            st.bar_chart(score_df.set_index("Class"), horizontal=True)
            st.dataframe(score_df, hide_index=True, use_container_width=True)
            export = {"filename": st.session_state.get("last_filename", "uploaded_xray"), "timestamp": datetime.now().isoformat(timespec="seconds"), "model_id": MODEL_ID, **result, "medical_disclaimer": "Educational output only; not a diagnosis."}
            st.download_button("Download result as JSON", data=json.dumps(export, indent=2), file_name="medvision_result.json", mime="application/json")
        else:
            st.markdown("### Your result will appear here")
            st.markdown('<div class="metric-card"><div class="metric-label">Prediction</div><div class="metric-value">Waiting for X-ray</div><p class="small-note">Upload an image and select Run X-ray analysis.</p></div>', unsafe_allow_html=True)
            st.markdown("**Supported model classes**")
            st.write("- NORMAL\n- PNEUMONIA")

elif page == "About the Model":
    st.subheader("About this classifier")
    st.write("This application uses a community-published pretrained image-classification model from Hugging Face. Model files are downloaded automatically on first use and cached by the Transformers library.")
    st.markdown(f"**Model:** [{MODEL_ID}](https://huggingface.co/{MODEL_ID})")
    st.markdown("**Output classes:** NORMAL and PNEUMONIA")
    st.markdown("**Input:** PNG, JPG, JPEG, WEBP, or BMP chest X-ray images")
    st.warning("Published benchmark results are reported by the model author and do not guarantee performance on your image, hospital, scanner, or population. This project is not a medical device.")

else:
    st.subheader("Run locally in VS Code")
    st.markdown(r"""
1. Install **Python 3.11 (64-bit)** for best compatibility with PyTorch.
2. Extract this ZIP and open the extracted folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate an environment:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
5. Start the app:
```powershell
streamlit run app.py
```
6. Open the local address printed in the terminal (usually `http://localhost:8501`). Upload a chest X-ray and click **Run X-ray analysis**.

**Troubleshooting:** The first model download needs internet and can take a few minutes. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. If the installation fails, use Python 3.11 rather than an experimental/new Python version.
""")
    st.info("No model training or local model file is required. The pretrained model is fetched on first run.")

st.divider()
st.caption("MedVision AI · Student portfolio project · Not for clinical use")
