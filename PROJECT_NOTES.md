# Project notes

- UI: Streamlit
- Image loading: Pillow
- Model runtime: PyTorch + Hugging Face Transformers
- Prediction: two-class chest X-ray classifier (NORMAL / PNEUMONIA)
- Inference model downloads and caches on first use
- No local training dataset or training step required
- Do not present outputs as clinical diagnosis

The source model card is community-published and reports evaluation on a dataset. This project does not independently validate the model.
