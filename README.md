# 🧪 Clustering Science Comments with K-Medoids and LDA

This project applies unsupervised learning techniques to cluster student comments into three predefined science topics: **Biology**, **Chemistry**, and **Physics**. We compare clustering performance using **K-Medoids** and **Latent Dirichlet Allocation (LDA)**, and benchmark them against supervised and random baselines.

---

## 📌 Objectives

- Group unlabeled science-related comments into meaningful topics using clustering
- Apply both **distance-based** (K-Medoids) and **probabilistic** (LDA) methods
- Compare against **supervised classification** and **random guessing** baselines
- Evaluate using standard metrics like **Silhouette Score**, **ARI**, **F1 Score**, and **Accuracy**

---

## 📂 Dataset Overview

- Comments labeled by topic: Biology, Chemistry, Physics
- Balanced across **training**, **validation**, and **test** sets
- Preprocessing includes:
  - Lowercasing
  - Removing punctuation
  - Stopword removal and stemming (for LDA)
  - Vectorization (TF-IDF, Word2Vec, CountVectorizer)

---

## 🧠 Models Implemented

### 🔹 K-Medoids Clustering
- Vectorized using TF-IDF and Word2Vec
- Grid search for optimal `n_clusters` and `max_features`
- Evaluated using Silhouette Score and Adjusted Rand Index

### 🔸 Latent Dirichlet Allocation (LDA)
- Implemented with both CountVectorizer and TF-IDF
- Grid search over:
  - Number of topics
  - Learning decay
  - Iterations
- Evaluated with Silhouette Score, ARI, F1 Score, Accuracy

### ⚖️ Baseline Models
- **Random classifier** using `DummyClassifier`
- **Supervised classifier** using `RandomForestClassifier`

---

## 📊 Results Summary

| Model                | Accuracy | F1 Score | Silhouette | ARI   |
|---------------------|----------|----------|-------------|-------|
| **Random**          | 33.35%   | 33.45%   | –           | –     |
| **Supervised (RF)** | 69.74%   | 69.74%   | –           | –     |
| **K-Medoids (TF-IDF)** | 35.30% | 33.09%   | 0.73        | ~0.00 |
| **LDA (CountVectorizer)** | 54.54% | 54.87%   | 0.52        | ~0.05 |

> 🔎 LDA outperformed K-Medoids in both accuracy and topic coherence, but was still far from supervised results.

---

## 🧪 Evaluation Metrics

- **Silhouette Score** – cohesion/separation of clusters
- **Adjusted Rand Index (ARI)** – alignment with true labels
- **F1 Score / Accuracy** – traditional performance measures


---

## ⚙️ Technologies Used

- Python 3
- `scikit-learn`, `pandas`, `nltk`, `gensim`, `matplotlib`
- Clustering: `KMedoids`, `LDA`
- Vectorization: `TF-IDF`, `Word2Vec`, `CountVectorizer`

---

## 📌 Key Takeaways

- Supervised models significantly outperform unsupervised ones, but require labeled data.
- LDA provided reasonable topic separation and interpretable results.
- K-Medoids performed poorly on semantic clustering despite good silhouette scores.

---

## 👨‍💻 Author

- [Mihai Dilirici](mailto:mihai.dilirici@s.unibuc.ro)

---

This repository is for educational and academic use. Contributions or feedback are welcome!

