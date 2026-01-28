# Hindi Sentiment Analysis using an Attention-Augmented CNN–BiLSTM Framework

This repository contains the implementation of an attention-augmented CNN–BiLSTM framework
for Hindi sentiment analysis using TF-IDF–weighted Word2Vec embeddings.
The code supports the experiments reported in the manuscript, including the proposed model
and an extensive ablation study.

## Repository Contents

- `Hindi_SA_Proposed_Model.py`  
  Implementation of the proposed attention-augmented CNN–BiLSTM architecture with
  TF-IDF-weighted Word2Vec embeddings for Hindi sentiment classification.

- `Ablation_study.py`  
  Scripts for conducting ablation experiments to evaluate the contribution of individual
  components such as the attention mechanism, TF-IDF weighting, CNN layers, BiLSTM,
  and regularization strategies.
  
- `Fine_tuned_transformer_direct_comparison.py`  
  Scripts for direct comparison with transformer-based baseline models, including
  multilingual BERT (mBERT) and XLM-RoBERTa (XLM-R), using identical training–testing splits
  and evaluation metrics as reported in the manuscript.
  
- `Statistical_Analysis_Hindi.py`  
  Script for performing paired statistical significance tests between the proposed model
  and baseline approaches using prediction outputs.

- `Augmented_Hinid_Product_Review_Dataset.csv`  
  The experiments initially utilize the Hindi Product Review Dataset released by IIT Patna, which contains 5,417 annotated
  review sentences and is publicly available via Kaggle and the official IIT Patna NLP
  resources page. To enhance data diversity and robustness, the dataset was further
  augmented by manually collecting additional Hindi product reviews from e-commerce
  platforms, primarily Flipkart. The final augmented dataset comprises 41,682 reviews, including 20,757 positive and 20,925   negative instances. Providing the more diverse benchmark for Hindi sentiment classification.
  
  The IIT Patna Hindi Product Review Dataset can be obtained from:
  - Kaggle: https://www.kaggle.com/datasets/warcoder/iit-patna-product-reviews/data
  - IIT Patna NLP Resources: http://www.iitp.ac.in/~ai-nlp-ml/resources.html

- `Acknowledgements`
  We acknowledge IIT Patna for providing the original Hindi Product Review Dataset. Additional reviews were collected from    Flipkart and manually curated exclusively for academic research and non-commercial use.


## Requirements

- Python 3.8 or higher
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Gensim
- Matplotlib
