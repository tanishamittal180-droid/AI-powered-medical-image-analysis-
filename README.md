# MedVision AI — Chest X-ray Classification Demo

A Streamlit portfolio application that uses a **pretrained, community-published image classifier** to return a chest X-ray class score for **NORMAL** or **PNEUMONIA**. It does not require you to train a model or provide a local checkpoint. On first use it downloads the model from Hugging Face and caches it locally.

> **Important:** This is for education and software demonstration only. It is not a medical device, does not diagnose patients, and must not be used to decide treatment. Scores are not calibrated clinical probabilities. Consult a qualified clinician for medical interpretation.

## Features

- Modern Streamlit interface with image preview
- Pretrained model loading and local cache
- NORMAL/PNEUMONIA classification and class-score visualization
- JSON result export
- Clear loading/error messages and project guide
- CPU inference supported; CUDA is used automatically when available

## Requirements

- Windows/macOS/Linux
- Python **3.11 64-bit recommended**
- Internet connection on first run to download model files
- Several GB of free disk space recommended for Python packages and cached weights

## Run on Windows (VS Code)

1. Extract the ZIP and open the extracted folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

4. Start the application:

```powershell
streamlit run app.py
```

5. Open the URL printed in the terminal, usually `http://localhost:8501`.
6. Choose **X-ray Analyzer**, upload a chest X-ray image, and click **Run X-ray analysis**.

## Model

- Model ID: [`nickmuchi/vit-finetuned-chest-xray-pneumonia`](https://huggingface.co/nickmuchi/vit-finetuned-chest-xray-pneumonia)
- Classes: NORMAL and PNEUMONIA
- Input: image files such as PNG/JPG/JPEG/WEBP/BMP
- The model files download at first use; an internet connection is required for that initial download.

The model card reports its own evaluation results; those figures are not a guarantee of real-world performance and may not generalize to other scanners, hospitals, populations, or image types. This binary model does **not** identify all diseases (for example, cancer, tuberculosis, COVID-19, edema, or fractures).

## Troubleshooting

- **`ModuleNotFoundError`:** activate `.venv` in the project folder and run `pip install -r requirements.txt` again.
- **Model download error:** check internet access, retry, and make sure Hugging Face is reachable. First download can take a while.
- **PyTorch installation errors:** use Python 3.11 64-bit, create a fresh virtual environment, and reinstall dependencies.
- **Port in use:** run `streamlit run app.py --server.port 8502`.
- **Blank page:** look at the terminal where Streamlit is running; the terminal error usually identifies the missing package or failure.

## Project structure

```text
AI-Medical-Image-Analysis/
├── app.py
├── requirements.txt
├── README.md
├── DISEASE_PREDICTION_SETUP.md
├── src/
│   ├── __init__.py
│   └── inference.py
├── models/                 # downloaded model is cached by Transformers
├── outputs/
└── docs/
```

## Responsible use

Only use de-identified images you are permitted to use. Do not upload patient names or other protected health information. The result is a machine-learning class output, not a diagnosis or substitute for a radiologist.
