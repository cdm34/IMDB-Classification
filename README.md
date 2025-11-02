# IMDB Sentiment Classification with LSTM  
**Author:** *Colin Adam*  
**Course:** CS 4420 – Artificial Intelligence, Spring 2025  
**Instructor:** Dr. Muchao Ye   

---

## Project Overview

This project implements a sentiment analysis classifier for IMDB movie reviews using an LSTM-based neural network in PyTorch. The goal is to classify reviews as positive or negative by preprocessing raw text, tokenizing and padding sequences, training an LSTM model, tuning hyperparameters, and evaluating the model using standard classification metrics.

This work was completed as part of **CS 4420** at the University of Iowa.

---

## Academic Attribution & Integrity

Portions of the codebase — specifically dataset loading, initial preprocessing functions, and the basic LSTM model template — were provided by Dr. Muchao Ye in the course tutorial (`text_classification_tutorial.ipynb`).  

I contributed by:

- Implementing model training, validation, and testing workflows  
- Performing hyperparameter search and model selection  
- Writing evaluation metrics (accuracy, precision, recall, F1, AUC)  
- Completing the IMDB sentiment classifier end-to-end based on project specifications  
- Writing final analysis and interpretation of results  

---

## Repository Structure

TODO

---

## Model Architecture

| Component        | Description |
|------------------|-------------|
| Embedding Layer  | Converts word indices into dense vectors |
| LSTM Layer       | Captures sequential structure in reviews |
| Fully Connected  | Maps LSTM hidden state → sentiment score |
| Output Activation| Sigmoid (binary classification) |

---

## Training & Hyperparameter Tuning

Two stages of hyperparameter search were conducted:

### Phase 1 – Broad Search
- Embedding sizes: 64, 128  
- Hidden sizes: 64, 128  
- Layers: 1, 2, 3  
- Learning rates: 0.1, 0.01, 0.001  
- Batch sizes: 16, 32, 64, 128  
- Epochs: 10  

### Phase 2 – Refined Search
- Embedding: 64, 128  
- Hidden: 64, 128  
- Layers: 1, 2  
- Learning Rate: 0.001  
- Batch Size: 16, 32  
- Epochs: 15  

Best validation accuracy achieved: 0.857

---

## Final Model Performance

| Metric        | Result |
|---------------|--------|
| Accuracy      | 0.849 |
| Precision     | 0.853 |
| Recall        | 0.844 |
| F1 Score      | 0.848 |
| AUC-ROC       | 0.921 |

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/cdm34/imdb-lstm-classifier.git
cd imdb-lstm-classifier
```

### 2. Create environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If requirements.txt isn’t available, manually install:
```bash
pip install torch torchvision nltk beautifulsoup4 numpy scikit-learn matplotlib
```

### 3. Download IMDB dataset
```bash
mkdir -p data
wget -O data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -zxf data/aclImdb_v1.tar.gz -C data
```

### 4. Run the Notebook
```bash
jupyter notebook imdb_lstm.ipynb
```
--- 

## Acknowledgements:
This project was developed as part of CS 4420: Artificial Inteligence at the University of Iowa.
Special thanks to Dr. Muchao Ye for providing the dataset loading code, preprocessing foundation, and project guidance.
