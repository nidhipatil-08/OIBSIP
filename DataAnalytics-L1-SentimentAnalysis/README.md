# Sentiment Analysis — Data Analytics Level 1, Task 4

![OIBSIP](https://img.shields.io/badge/OIBSIP-Level%201-blue)
![Python](https://img.shields.io/badge/Python-3.13-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project Overview

This project is part of the **Oasis Infobyte Internship Program (OIBSIP)** — **Data Analytics, Level 1, Task 4**. The objective is to build a machine learning model that classifies the sentiment of text data (positive, negative, or neutral) and to derive insights into public opinion or customer feedback.

## Objective

- Load and inspect a text-based sentiment dataset
- Build a complete text preprocessing pipeline (lowercase, punctuation removal, tokenisation, stopword removal, lemmatisation)
- Convert cleaned text into numerical features using TF-IDF
- Train and compare at least two classifiers: **Multinomial Naive Bayes** and **Linear SVM**
- Evaluate models with accuracy, precision, recall, F1-score, and confusion matrices
- Visualise the data with a class distribution bar chart and WordClouds per sentiment
- Analyse 5 misclassified examples and discuss why they failed
- Provide a conclusion identifying the best model and a real-world application

## Repository Structure
DataAnalytics-L1-SentimentAnalysis/
│
├── IMDB Dataset.csv # Raw dataset (50,000 movie reviews)
├── Source Code.ipynb # Jupyter/Colab notebook with the full pipeline
└── README.md # Project documentation 


## Dataset

- **Source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Records:** 50,000 movie reviews
- **Classes:** Positive / Negative (perfectly balanced — 25,000 each)
- **Missing Values:** None
- **Columns:**

| Column | Description |
|---|---|
| review | Raw movie review text |
| sentiment | Target label — "positive" or "negative" |

> **Note on the neutral class:** The IMDB dataset is binary. To satisfy the task's requirement of inspecting positive/negative/neutral counts, a third "neutral" class was derived using **TextBlob's polarity score** on a 5,000-review sample.

## 🛠️ Tools & Libraries

- **Python 3.10**
- **pandas**, **numpy** — data handling
- **NLTK** — stopword removal and lemmatisation
- **scikit-learn** — TF-IDF vectorisation, Naive Bayes, Linear SVM, evaluation metrics
- **matplotlib**, **seaborn** — visualisations
- **WordCloud** — sentiment word clouds
- **TextBlob** — rule-based polarity baseline and neutral class derivation
- **Google Colab** — development environment

## Analysis Workflow

### 1. Data Loading & Inspection
- Loaded the IMDB dataset into a pandas DataFrame
- Checked shape, column names, class distribution, and missing values
- Confirmed a **perfectly balanced binary dataset** (25,000 positive / 25,000 negative)

### 2. Text Preprocessing Pipeline
Applied a five-step pipeline to every review:
1. **Lowercasing** — normalises "Good" and "good"
2. **HTML tag removal** — strips `<br />` artifacts present in IMDB reviews
3. **Punctuation & number removal** — removes non-alphabetic characters
4. **Tokenisation** — splits text into individual words
5. **Stopword removal & lemmatisation** — drops common words and reduces to base forms (e.g., "running" → "run")

### 3. Feature Extraction — TF-IDF
Converted cleaned text into a sparse numeric matrix using **TF-IDF with 5,000 features**. TF-IDF down-weights words that appear everywhere (e.g., "the") and highlights distinctive sentiment-bearing words (e.g., "brilliant", "disappointing").

### 4. Train/Test Split
- 80/20 stratified split
- 40,000 training samples, 10,000 test samples
- Class balance preserved in both sets

### 5. Model Training
Two classifiers were trained and compared:
- **Multinomial Naive Bayes** — probabilistic baseline for text classification
- **Linear SVM (LinearSVC)** — linear support vector classifier, well suited for high-dimensional sparse TF-IDF vectors

### 6. Evaluation
For each model: accuracy, precision, recall, F1-score, and confusion matrix.

### 7. Visualisation
- Sentiment distribution bar chart
- WordCloud for positive reviews and negative reviews
- 3-class (positive/neutral/negative) distribution via TextBlob

### 8. Error Analysis
Displayed 5 misclassified examples from the Naive Bayes model with discussion of failure modes.

## Key Results

| Metric | Naive Bayes | Linear SVM |
|---|---|---|
| **Accuracy** | 0.8525 | **0.8771** |
| **Precision** | 0.8477 | **0.8726** |
| **Recall** | 0.8594 | **0.8832** |
| **F1-Score** | 0.8535 | **0.8778** |

**Best model:** Linear SVM outperformed Naive Bayes across all metrics with ~**87.7% accuracy** on the test set.

### 3-Class Sentiment via TextBlob (on 5,000-review sample)
- Positive: ~50.5%
- Neutral: ~39.5%
- Negative: ~10.0%

### Error Analysis — Common Failure Modes
1. **Sarcasm** — positive words used ironically confuse the bag-of-words model
2. **Mixed sentiment** — reviews praising and criticising in the same sentence
3. **Negation** — "Not bad at all" contains "bad" but means positive
4. **Very short text** — insufficient signal to classify confidently
5. **Domain-specific language** — "so bad it's good", "cult classic" phrases

## Business / Real-World Applications

- Analysing customer product reviews to detect dissatisfaction early
- Monitoring brand sentiment across social media platforms
- Triaging customer support tickets by emotional urgency
- Gauging public opinion on product launches or policy announcements
- Filtering user feedback in app stores by sentiment

## Limitations

- Bag-of-words models cannot detect sarcasm, irony, or negation scope
- Word order and context are lost during vectorisation
- The IMDB dataset is binary; the neutral class was rule-derived, not natively labelled

## Future Improvements

- Replace TF-IDF with word embeddings (Word2Vec, GloVe)
- Fine-tune a transformer model such as BERT for context-aware sentiment
- Add a neutral class using a semi-supervised approach
- Increase training data or apply data augmentation

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/OIBSIP.git
   cd OIBSIP/DataAnalytics-L1-SentimentAnalysis

2. Install dependencies (in Google Colab, most are pre-installed):
   pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud textblob

3. Open the notebook:

   Locally: jupyter notebook Source Code.ipynb
   On Colab: Upload Source Code.ipynb and IMDB Dataset.csv to the same runtime

4. Run all cells in order.

📸 Visualizations Included
Sentiment Distribution Bar Chart

Confusion Matrices (Naive Bayes + Linear SVM)

WordCloud — Positive Reviews

WordCloud — Negative Reviews

3-Class Distribution Chart (TextBlob)

Model Comparison Chart

Author
Nidhi Patil
Data Analytics Intern — Oasis Infobyte (OIBSIP)
GitHub: @nidhipatil-08

Acknowledgements
Oasis Infobyte for the internship opportunity and project guidelines
Kaggle for hosting the IMDB 50K Movie Reviews dataset
NLTK, scikit-learn, and TextBlob maintainers for the open-source tooling
