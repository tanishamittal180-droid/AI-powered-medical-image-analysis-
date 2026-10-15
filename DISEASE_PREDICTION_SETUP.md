# Disease-class output setup

This version is wired to a public pretrained model so you do not have to train a CNN yourself.

## What it can predict

- `NORMAL`
- `PNEUMONIA`

It cannot reliably recognize every disease or confirm whether a person is ill. The displayed score is the model's softmax output, not a calibrated clinical probability.

## Install and run

Use Python 3.11 64-bit, then in VS Code Terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On the first analysis, an internet connection is needed to download the model weights from Hugging Face. Keep the terminal open while Streamlit runs. If model loading fails, the application shows an error and technical details.

## Model reference

https://huggingface.co/nickmuchi/vit-finetuned-chest-xray-pneumonia

This is a community model for educational demonstration, not a clinically validated diagnostic tool.
