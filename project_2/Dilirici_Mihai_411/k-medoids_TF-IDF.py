import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer


# Load the datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Split train.csv into training and validation sets
train_comments, val_comments, train_labels, val_labels = train_test_split(
    df_train['Comment'], df_train['Topic'], test_size=0.2, random_state=42
)

df_train_split = pd.DataFrame({'Comment': train_comments, 'Topic': train_labels})
df_val_split = pd.DataFrame({'Comment': val_comments, 'Topic': val_labels})

# Define a function to perform grid search
def grid_search_kmedoids(data, labels, cluster_range, max_features_range):
    best_score = float('-inf')
    best_params = None
    results = []

    for max_features in max_features_range:
        # Vectorize text using TF-IDF with varying max_features
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = vectorizer.fit_transform(data).toarray()

        # Normalize the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Reduce dimensionality
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        for n_clusters in cluster_range:
            # Apply K-Medoids clustering
            kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
            kmedoids.fit(X_reduced)

            # Check if at least two clusters are formed
            if len(set(kmedoids.labels_)) > 1:
                sil_score = silhouette_score(X_reduced, kmedoids.labels_)
                ari_score = adjusted_rand_score(labels, kmedoids.labels_)
                results.append((n_clusters, max_features, sil_score, ari_score))

                print(f"n_clusters: {n_clusters}, max_features: {max_features}, Silhouette Score: {sil_score}, ARI: {ari_score}")

                # Update best parameters if a better score is found
                if sil_score > best_score:
                    best_score = sil_score
                    best_params = {'n_clusters': n_clusters, 'max_features': max_features}

            else:
                print(f"n_clusters: {n_clusters}, max_features: {max_features} - Only one cluster found.")

    return best_params, best_score, results

# Perform the grid search on training data
cluster_range = range(3, 6)  # Test 3 to 5 clusters
max_features_range = [300, 400, 500, 700]  # Test different feature sizes for TF-IDF


best_params, best_score, grid_results = grid_search_kmedoids(
    df_train_split['Comment'], df_train_split['Topic'], cluster_range, max_features_range
)

print(f'\nBest Parameters: {best_params}')
print(f'Best Silhouette Score: {best_score}')

# Step 4: Visualize the grid search results
results_df = pd.DataFrame(grid_results, columns=['n_clusters', 'max_features', 'silhouette_score', 'ari'])

# Step 5: Apply K-Medoids with the best parameters
vectorizer = TfidfVectorizer(stop_words='english', max_features=best_params['max_features'])
X_train = vectorizer.fit_transform(df_train_split['Comment']).toarray()
X_val = vectorizer.transform(df_val_split['Comment']).toarray()
X_test = vectorizer.transform(df_test['Comment']).toarray()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_val_reduced = pca.transform(X_val_scaled)
X_test_reduced = pca.transform(X_test_scaled)

kmedoids = KMedoids(n_clusters=best_params['n_clusters'], random_state=42)
kmedoids.fit(X_train_reduced)

# Add the cluster labels to the training and validation DataFrames
df_train_split['Cluster'] = kmedoids.labels_
df_val_split['Cluster'] = kmedoids.predict(X_val_reduced)

# Step 6: Evaluate the clustering on the validation set
if len(set(df_val_split['Cluster'])) > 1:
    sil_score_val = silhouette_score(X_val_reduced, df_val_split['Cluster'])
    ari_score_val = adjusted_rand_score(df_val_split['Topic'], df_val_split['Cluster'])
    print(f'Validation Silhouette Score: {sil_score_val}')
    print(f'Validation ARI: {ari_score_val}')
else:
    print("Validation metrics cannot be calculated with only one cluster.")

# Apply the clustering model to the test data
df_test['Cluster'] = kmedoids.predict(X_test_reduced)

# Step 7: Evaluate the clustering on the test set
if len(set(df_test['Cluster'])) > 1:
    sil_score_test = silhouette_score(X_test_reduced, df_test['Cluster'])
    ari_score_test = adjusted_rand_score(df_test['Topic'], df_test['Cluster'])
    print(f'Test Silhouette Score: {sil_score_test}')
    print(f'Test ARI: {ari_score_test}')
else:
    print("Test metrics cannot be calculated with only one cluster.")