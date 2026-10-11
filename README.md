# Restaurant Recommendation System Using Document Embeddings

A text-based restaurant recommendation system that predicts user ratings from Yelp reviews using Word2Vec, FastText, and Doc2Vec embeddings. The project explores document representation techniques, regression-based rating prediction, and hyperparameter optimization.

## Overview

The system converts restaurant review text into numerical document representations and uses these representations as input features for regression models.

The workflow includes:

- Review data preprocessing and filtering
- Word embedding training with Word2Vec and FastText
- Document representation using pooling strategies and Doc2Vec
- Regression model training and comparison
- Hyperparameter optimization with Optuna
- Model evaluation using regression and ranking metrics
- Restaurant recommendation based on predicted ratings

## Methodology

### Text Preprocessing

Review text is converted to lowercase and tokenized using NLTK. The processed reviews are used to train embedding models and construct document-level representations.

### Document Embeddings

The project implements and compares the following approaches:

| Model | Configuration | Representation |
|---|---|---|
| Word2Vec | Skip-gram (SGNS) | Mean, sum, and max pooling |
| Word2Vec | CBOW | Mean, sum, and max pooling |
| FastText | Skip-gram (SGNS) | Mean, sum, and max pooling |
| FastText | CBOW | Mean, sum, and max pooling |
| Doc2Vec | Distributed Memory (DM) | Learned document vectors |
| Doc2Vec | Distributed Bag of Words (DBOW) | Learned document vectors |

Word2Vec and FastText learn word-level embeddings from contextual information. Pooling operations aggregate word vectors into review representations, while Doc2Vec learns document-level embeddings directly.

### Rating Prediction

Document embeddings are used as input features for supervised regression models. The initial comparison includes:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- K-Nearest Neighbors Regressor

Optuna is used to optimize hyperparameters across the embedding configurations.

### Evaluation Metrics

The implementation evaluates rating prediction and recommendation ranking using the following metrics.

**Regression metrics**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)
- Concordance Correlation Coefficient (CCC)
- Pearson correlation coefficient

**Ranking metrics**
- Normalized Discounted Cumulative Gain (NDCG) at different cutoffs
- Spearman's rank correlation coefficient
- Kendall's rank correlation coefficient

## Tech Stack

- **Python** — Core implementation
- **Pandas and NumPy** — Data processing
- **NLTK** — Text tokenization
- **Gensim** — Word2Vec, FastText, and Doc2Vec
- **Scikit-learn** — Regression models and evaluation
- **Optuna** — Hyperparameter optimization
- **Matplotlib and Seaborn** — Visualization
- **Jupyter Notebook** — Experimentation and analysis

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install pandas numpy nltk gensim scikit-learn optuna matplotlib seaborn jupyter
```

## Dataset Preparation

The project uses the Yelp Open Dataset. Download the dataset separately and configure the input paths in the notebook to point to the extracted business and review JSON files.

The preprocessing pipeline:

1. Selects businesses categorized as restaurants.
2. Identifies users who have reviewed at least 100 distinct restaurants.
3. Identifies restaurants reviewed by at least 1,000 unique users.
4. Filters reviews to retain records associated with the selected users and restaurants.
5. Sorts reviews chronologically and creates training, validation, and test splits.
6. Saves the processed splits as line-delimited JSON files.

Update all input and output paths to match your local environment before running the notebook.

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the project's main notebook and execute the cells in order. Run the preprocessing pipeline before training the embedding models, since subsequent experiments depend on the processed review data.

The notebook covers the complete workflow, from data preparation and embedding generation to regression training, hyperparameter optimization, evaluation, and recommendation generation.

## Reproducibility

For consistent experiments:

- Use the same preprocessing procedure across embedding configurations.
- Keep random seeds fixed where supported.
- Use the training set for fitting models and the validation set for model selection and hyperparameter tuning.
- Reserve the test set for final evaluation.
- Record model configurations and evaluation metrics when comparing experiments.