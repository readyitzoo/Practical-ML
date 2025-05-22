import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr, kendalltau
from xgboost import XGBRegressor
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from math import sqrt
import matplotlib.pyplot as plt

# nltk.download("wordnet")
# nltk.download("stopwords")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Custom Spearman correlation scorer
def spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

# Custom Kendall's Tau scorer
def kendall_corr(y_true, y_pred):
    return kendalltau(y_true, y_pred).correlation

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load datasets
train_file = "train.csv"
val_file = "val.csv"
test_file = "test.csv"

train_data = load_data(train_file)
val_data = load_data(val_file)
test_data = load_data(test_file)

# Preprocess text data for word2vec
def preprocess_text(text):
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize, lemmatize, and remove stopwords
    tokens = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.lower() not in stop_words]
    return tokens

# Combine text data from training, validation, and test sets into one list
train_sentences = pd.concat([train_data["text"], val_data["text"], test_data["text"]])

# Preprocess the combined sentences (e.g., tokenize, lemmatize, and clean)
combined_texts = train_sentences.apply(preprocess_text)

# Train Word2Vec model on the combined dataset
model_w2v = Word2Vec(sentences=combined_texts, vector_size=150, window=5, min_count=1, workers=5, epochs=80, seed=55)
model_w2v.save("word2vec.model")

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=700)
tfidf.fit(train_sentences.apply(lambda x: " ".join(preprocess_text(x))))

# Create word2vec embeddings for each document
def get_combined_features(text, model_w2v, tfidf_vocab):
    tfidf_scores = {word: tfidf_vocab[word] for word in text if word in tfidf_vocab}
    # Word2Vec embeddings for words in the document
    embeddings = [model_w2v.wv[word] * tfidf_scores.get(word, 0) for word in text if word in model_w2v.wv]
    if len(embeddings) == 0:
        return np.zeros(model_w2v.vector_size)
    # Return normalized combined vector
    return normalize(np.mean(embeddings, axis=0).reshape(1, -1)).flatten()

# Extract TF-IDF vocabulary
tfidf_vocab = tfidf.vocabulary_

# Apply combined features to training, validation, and test sets
X_train = np.array([get_combined_features(preprocess_text(text), model_w2v, tfidf_vocab) for text in train_data["text"]])
X_val = np.array([get_combined_features(preprocess_text(text), model_w2v, tfidf_vocab) for text in val_data["text"]])
X_test = np.array([get_combined_features(preprocess_text(text), model_w2v, tfidf_vocab) for text in test_data["text"]])

# Targets
y_train = train_data["score"].values
y_val = val_data["score"].values

# # Parameter grid for random search
# param_grid = {
#     "learning_rate": [0.01, 0.03],
#     "max_depth": [7, 10],
#     "n_estimators": [500],
#     "subsample": [0.7, 1.0],
#     "colsample_bytree": [0.8],
#     "gamma": [0.3, 0.5, 1],
#     "lambda": [0.5, 0.8, 1],
#     "alpha": [1, 2, 5, 10],
# }

# Initialize XGBRegressor with specific parameters
xgb_reg = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    learning_rate=0.003,
    max_depth=6,
    n_estimators=300,
    subsample=0.9,
    colsample_bytree=0.6,
    lambda_=0.5,
    alpha=1
)

# Train the model without eval_metric
xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

# Get the number of epochs (iterations)
epochs = xgb_reg.n_estimators
x_axis = range(0, epochs)

# Initialize lists to store RMSE values for train and validation sets
train_rmse = []
val_rmse = []

# Loop through each epoch to calculate RMSE
for i in range(epochs):
    # Get predictions for the current iteration
    y_train_pred = xgb_reg.predict(X_train, iteration_range=(0, i+1))
    y_val_pred = xgb_reg.predict(X_val, iteration_range=(0, i+1))

    # Calculate RMSE for train and validation
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    val_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))

# Plotting the learning curve for RMSE
fig, ax = plt.subplots()
ax.plot(x_axis, train_rmse, label='Train RMSE')
ax.plot(x_axis, val_rmse, label='Validation RMSE')
ax.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('RMSE')
plt.title('XGBoost RMSE Learning Curve')
plt.show()
# Evaluate the model on the validation data
y_val_pred = xgb_reg.predict(X_val)

# Calculate the evaluation metrics
mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
rmse = sqrt(mse)  # Calculate RMSE
spearman_corr_value = spearman_corr(y_val, y_val_pred)
kendall_corr_value = kendall_corr(y_val, y_val_pred)

# Print all evaluation metrics
print(f"Validation MAE: {mae:.4f}")
print(f"Validation MSE: {mse:.4f}")
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation Spearman's Rank Correlation Coefficient: {spearman_corr_value:.4f}")
print(f"Validation Kendall's Tau: {kendall_corr_value:.4f}")

# Create submission file
def create_submission_file(model, X_test, test_ids, output_file):
    predictions = model.predict(X_test)
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "score": np.round(predictions, 8),
        }
    )
    submission.to_csv(output_file, index=False)

# Generate submission file
submission_file = "submission.csv"
create_submission_file(xgb_reg, X_test, test_data["id"], submission_file)

