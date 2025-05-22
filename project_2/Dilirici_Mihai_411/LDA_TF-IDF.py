import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Load the datasets
train_file_path = 'train.csv'
test_file_path = 'test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Preprocessing function
def preprocess_text(text):
    # Text preprocessing: Lowercasing, removing punctuation, removing stopwords, and stemming.
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords and stem the words
    text = " ".join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to the 'Comment' column for both train and test datasets
train_data['processed_comment'] = train_data['Comment'].apply(preprocess_text)
test_data['processed_comment'] = test_data['Comment'].apply(preprocess_text)

# Split train data into train and validation
train_data, val_data = train_test_split(
    train_data,
    test_size=0.2,
    random_state=42,
    stratify=train_data['Topic'] if 'Topic' in train_data.columns else None
)

# Define parameter grid
param_grid = {
    'max_features': [300, 500],
    'n_components': [3, 5],
    'learning_decay': [0.3],
    'max_iter': [10, 20, 30]
}

# Custom grid search
best_silhouette_score = float('-inf')
best_params = None

for max_features in param_grid['max_features']:
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_train = vectorizer.fit_transform(train_data['processed_comment'])
    X_val = vectorizer.transform(val_data['processed_comment'])
    X_test = vectorizer.transform(test_data['processed_comment'])

    for n_components in param_grid['n_components']:
        for learning_decay in param_grid['learning_decay']:
            for max_iter in param_grid['max_iter']:
                # Define and fit LDA model
                lda = LatentDirichletAllocation(
                    n_components=n_components,
                    learning_decay=learning_decay,
                    max_iter=max_iter,
                    random_state=42
                )
                lda.fit(X_train)

                # Evaluate on validation data
                lda_topic_distributions_val = lda.transform(X_val)
                lda_labels_val = np.argmax(lda_topic_distributions_val, axis=1)

                # Calculate Silhouette Score and ARI
                if len(set(lda_labels_val)) > 1:
                    sil_score_val = silhouette_score(lda_topic_distributions_val, lda_labels_val)
                    ari_score_val = adjusted_rand_score(val_data['Topic'], lda_labels_val)
                else:
                    sil_score_val = float('-inf')
                    ari_score_val = float('-inf')

                print(f"Max Features: {max_features}, N Components: {n_components}, "
                      f"Learning Decay: {learning_decay}, Max Iter: {max_iter}")
                print(f"Validation Silhouette Score: {sil_score_val:.4f}, Validation ARI: {ari_score_val:.4f}")

                # Save best parameters based on Silhouette Score
                if sil_score_val > best_silhouette_score:
                    best_silhouette_score = sil_score_val
                    best_params = {
                        'max_features': max_features,
                        'n_components': n_components,
                        'learning_decay': learning_decay,
                        'max_iter': max_iter
                    }

# Print best parameters and Silhouette Score
print("\nBest Parameters:", best_params)
print("Best Validation Silhouette Score:", best_silhouette_score)

# Save the test results with the best parameters
vectorizer = TfidfVectorizer(stop_words='english', max_features=best_params['max_features'])
X_train = vectorizer.fit_transform(train_data['processed_comment'])
X_test = vectorizer.transform(test_data['processed_comment'])

lda = LatentDirichletAllocation(
    n_components=best_params['n_components'],
    learning_decay=best_params['learning_decay'],
    max_iter=best_params['max_iter'],
    random_state=42
)
lda.fit(X_train)

lda_topic_distributions_test = lda.transform(X_test)
lda_labels_test = np.argmax(lda_topic_distributions_test, axis=1)

# Silhouette Score and ARI for Test Data
if len(set(lda_labels_test)) > 1:
    sil_score_test = silhouette_score(lda_topic_distributions_test, lda_labels_test)
    ari_score_test = adjusted_rand_score(test_data['Topic'], lda_labels_test)
    print(f"\nTest Silhouette Score: {sil_score_test:.4f}")
    print(f"Test Adjusted Rand Index (ARI): {ari_score_test:.4f}")
else:
    print("\nTest Silhouette Score and ARI cannot be calculated with only one cluster.")

from sklearn.metrics import f1_score, accuracy_score

# Ensure ground truth labels are numeric
def ensure_numeric_labels(labels):
    if labels.dtype == 'object':
        unique_labels = labels.unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        return labels.map(label_mapping).values, label_mapping
    return labels.values, None

# Get numeric labels for validation and test datasets
val_ground_truth, val_mapping = ensure_numeric_labels(val_data['Topic'])
test_ground_truth, test_mapping = ensure_numeric_labels(test_data['Topic'])

# Calculate F1 Score and Accuracy for validation data
f1_val = f1_score(val_ground_truth, lda_labels_val, average='weighted')
accuracy_val = accuracy_score(val_ground_truth, lda_labels_val)

print(f"Validation F1 Score: {f1_val:.4f}")
print(f"Validation Accuracy: {accuracy_val:.4f}")

# Calculate F1 Score and Accuracy for test data
f1_test = f1_score(test_ground_truth, lda_labels_test, average='weighted')
accuracy_test = accuracy_score(test_ground_truth, lda_labels_test)

print(f"Test F1 Score: {f1_test:.4f}")
print(f"Test Accuracy: {accuracy_test:.4f}")