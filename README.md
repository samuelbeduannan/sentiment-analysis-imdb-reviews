# Sentiment Analysis on IMDB Dataset Using Support Vector Machine

## Project Description

This project focuses on sentiment analysis of IMDB movie reviews using a **Support Vector Machine (SVM)** model. The goal is to classify reviews as positive or negative based on textual content.

### Why Use SVM Instead of Other Models?

1. **Logistic Regression** – Weak at capturing complex patterns in text data.
2. **Naïve Bayes** – Struggles to understand contextual meaning in text.
3. **Random Forest** – Performs poorly on text classification tasks compared to SVM.

By leveraging **AWS training jobs** and **MLflow**, I was able to efficiently train, experiment, and track my model's performance using key metrics.

### Model Performance

#### Training Metrics:

- **Precision:** 0.9469
- **Recall:** 0.9468
- **F1 Score:** 0.9469
- **Accuracy:** 0.9469

#### Validation Metrics:

- **Accuracy:** 0.8934
- **F1 Score:** 0.8934
- **Precision:** 0.8936
- **Recall:** 0.8933

---

## Folder Structure

```
├── dataset/
│   ├── imdb_reviews.csv  # Raw IMDB dataset
│
├── training_code/
│   ├── train.py  # Training script for sentiment analysis
│   ├── requirements.txt  # Dependencies for training
│
├── model/
│   ├── sentiment_svm.pkl  # Trained SVM model
│   ├── vectorizer.pkl  # TF-IDF vectorizer
│
├── notebooks/
│   ├── sentiment_analysis.ipynb  # Jupyter notebook for model training and evaluation
│
├── README.md  # Project documentation
```

---

## Jupyter Notebook Breakdown

The **notebooks/sentiment_analysis.ipynb** contains:

1. **Data Loading & Preparation** – Cleaning and preprocessing the dataset.
2. **Data Exploration** –
   - Bar graphs showing positive/negative sentiment distribution.
   - WordCloud visualization for common words in reviews.
3. **Model Engineering** –
   - Using `sklearn`’s **SVM** model.
   - Tracking experiments with **MLflow**.
4. **Model Training** –
   - Using **SageMaker Estimator** to train the model on AWS.
5. **Model Evaluation** – Assessing model performance using precision, recall, F1-score, and accuracy.
6. **Model Deployment** – Deploying the trained model to an endpoint.

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r training_code/requirements.txt
```

### 2. Train the Model

```bash
python training_code/train.py
```

### 3. Run Jupyter Notebook

```bash
jupyter notebook
```

Open `notebooks/sentiment_analysis.ipynb` and follow the steps.

---

## Tools & Technologies

- **Python (scikit-learn, pandas, numpy, matplotlib, seaborn)**
- **AWS SageMaker (for training and deployment)**
- **MLflow (for experiment tracking)**
- **Jupyter Notebook (for analysis and visualization)**

---

## Acknowledgments

This project is inspired by the need for efficient sentiment classification in movie reviews. **SVM** was chosen due to its superior performance in text classification compared to other traditional models.
