from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

# Use the same TF-IDF vectorizer for all datasets
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Consistent max_features

# Fit on train data and transform train, validation, and test data
X_train = tfidf_vectorizer.fit_transform(train_data['processed_comment'])
X_val = tfidf_vectorizer.transform(val_data['processed_comment'])
X_test = tfidf_vectorizer.transform(test_data['processed_comment'])

# Create a dummy classifier that predicts random classes
dummy_clf = DummyClassifier(strategy='uniform', random_state=42)
dummy_clf.fit(X_train, train_data['Topic'])  # Use training data for fitting

# Random baseline predictions
random_val_predictions = dummy_clf.predict(X_val)
random_test_predictions = dummy_clf.predict(X_test)

# Evaluate random baseline
random_val_accuracy = accuracy_score(val_data['Topic'], random_val_predictions)
random_test_accuracy = accuracy_score(test_data['Topic'], random_test_predictions)

random_val_f1 = f1_score(val_data['Topic'], random_val_predictions, average='weighted')
random_test_f1 = f1_score(test_data['Topic'], random_test_predictions, average='weighted')

print(f"Random Baseline Validation Accuracy: {random_val_accuracy}")
print(f"Random Baseline Test Accuracy: {random_test_accuracy}")
print(f"Random Baseline Validation F1 Score: {random_val_f1}")
print(f"Random Baseline Test F1 Score: {random_test_f1}")

# Train a supervised model (Random Forest Classifier) on the TF-IDF-transformed training data
supervised_clf = RandomForestClassifier(random_state=42)
supervised_clf.fit(X_train, train_data['Topic'])

# Predict using the supervised model
supervised_val_predictions = supervised_clf.predict(X_val)
supervised_test_predictions = supervised_clf.predict(X_test)

# Evaluate supervised baseline
supervised_val_accuracy = accuracy_score(val_data['Topic'], supervised_val_predictions)
supervised_test_accuracy = accuracy_score(test_data['Topic'], supervised_test_predictions)

supervised_val_f1 = f1_score(val_data['Topic'], supervised_val_predictions, average='weighted')
supervised_test_f1 = f1_score(test_data['Topic'], supervised_test_predictions, average='weighted')

print(f"Supervised Baseline Validation Accuracy: {supervised_val_accuracy}")
print(f"Supervised Baseline Test Accuracy: {supervised_test_accuracy}")
print(f"Supervised Baseline Validation F1 Score: {supervised_val_f1}")
print(f"Supervised Baseline Test F1 Score: {supervised_test_f1}")