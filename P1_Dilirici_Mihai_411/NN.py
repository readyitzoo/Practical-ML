import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

# Extract features and targets
combined_text = pd.concat([train_data['text'], val_data['text']])
tfidf = TfidfVectorizer(max_features=750)
tfidf.fit(combined_text)

X_train = tfidf.transform(train_data['text']).toarray()
y_train = train_data['score']

X_val = tfidf.transform(val_data['text']).toarray()
y_val = val_data['score']

X_test = tfidf.transform(test_data['text']).toarray()

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Start with 50 epochs and adjust as needed
    batch_size=16,
    verbose=1
)

# Evaluate the model
y_val_pred = model.predict(X_val).flatten()
val_correlation = spearman_corr(y_val, y_val_pred)

# Calculate the evaluation metrics
mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
spearman_corr_value = spearman_corr(y_val, y_val_pred)
kendall_corr_value = kendall_corr(y_val, y_val_pred)

# Print all evaluation metrics
print(f"Validation MAE: {mae:.4f}")
print(f"Validation MSE: {mse:.4f}")
print(f"Validation Spearman's Rank Correlation Coefficient: {spearman_corr_value:.4f}")
print(f"Validation Kendall's Tau: {kendall_corr_value:.4f}")
# Create submission file
def create_submission_file(model, test_features, test_ids, output_file):
    predictions = model.predict(test_features).flatten()
    submission = pd.DataFrame({
        'id': test_ids,
        'score': np.round(predictions, 8)
    })
    submission.to_csv(output_file, index=False)

# Generate submission file
submission_file = "submission.csv"
create_submission_file(model, X_test, test_data['id'], submission_file)
