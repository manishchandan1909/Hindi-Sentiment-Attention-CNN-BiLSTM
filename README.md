# Hindi Sentiment Analysis using an Attention-Augmented CNN–BiLSTM Framework

This repository contains the implementation of an attention-augmented CNN–BiLSTM framework
for Hindi sentiment analysis using TF-IDF–weighted Word2Vec embeddings.
The code supports the experiments reported in the manuscript, including the proposed model
and an extensive ablation study.

## Repository Contents

- `proposed_model.py`  
  Implementation of the proposed attention-augmented CNN–BiLSTM architecture with
  TF-IDF-weighted Word2Vec embeddings for Hindi sentiment classification.

- `ablation_study.py`  
  Scripts for conducting ablation experiments to evaluate the contribution of individual
  components such as the attention mechanism, TF-IDF weighting, CNN layers, BiLSTM,
  and regularization strategies.

- `data/`  
  Placeholder directory for datasets. Due to licensing and distribution constraints,
  the dataset is not included in this repository.

## Requirements

- Python 3.8 or higher
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Gensim
- Matplotlib

Install all dependencies using:
```bash
pip install -r requirements.txt

