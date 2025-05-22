import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np

# Load the datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Split train.csv into training and validation sets
train_comments, val_comments, train_labels, val_labels = train_test_split(
    df_train['Comment'], df_train['Topic'], test_size=0.2, random_state=42
)

df_train_split = pd.DataFrame({'Comment': train_comments, 'Topic': train_labels})
df_val_split = pd.DataFrame({'Comment': val_comments, 'Topic': val_labels})

# Preprocessing function to tokenize text
def preprocess_and_tokenize(text):
    return text.lower().split()

df_train_split['Tokenized_Comment'] = df_train_split['Comment'].apply(preprocess_and_tokenize)
df_val_split['Tokenized_Comment'] = df_val_split['Comment'].apply(preprocess_and_tokenize)
df_test['Tokenized_Comment'] = df_test['Comment'].apply(preprocess_and_tokenize)

# Step 1: Train a Word2Vec model
word2vec_model = Word2Vec(sentences=df_train_split['Tokenized_Comment'], vector_size=300, window=5, min_count=1, workers=4, seed=42)

# Step 2: Generate sentence embeddings (average of word vectors)
def compute_sentence_embedding(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if len(vectors) == 0:  # Handle cases where no word in the sentence has a Word2Vec embedding
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Generate embeddings for train, validation, and test sets
def generate_embeddings(data, model):
    return np.array([compute_sentence_embedding(sentence, model) for sentence in data])

X_train = generate_embeddings(df_train_split['Tokenized_Comment'], word2vec_model)
X_val = generate_embeddings(df_val_split['Tokenized_Comment'], word2vec_model)
X_test = generate_embeddings(df_test['Tokenized_Comment'], word2vec_model)

# Step 3: Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 4: Reduce dimensionality
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_val_reduced = pca.transform(X_val_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Step 5: Apply K-Medoids clustering
cluster_range = range(3, 6)

def grid_search_kmedoids(data, labels, cluster_range):
    best_score = float('-inf')
    best_params = None
    results = []

    for n_clusters in cluster_range:
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
        kmedoids.fit(data)

        if len(set(kmedoids.labels_)) > 1:
            sil_score = silhouette_score(data, kmedoids.labels_)
            ari_score = adjusted_rand_score(labels, kmedoids.labels_)
            results.append((n_clusters, sil_score, ari_score))

            print(f"n_clusters: {n_clusters}, Silhouette Score: {sil_score}, ARI: {ari_score}")

            if sil_score > best_score:
                best_score = sil_score
                best_params = {'n_clusters': n_clusters}
        else:
            print(f"n_clusters: {n_clusters} - Only one cluster found.")

    return best_params, best_score, results

best_params, best_score, grid_results = grid_search_kmedoids(X_train_reduced, df_train_split['Topic'], cluster_range)

print(f'\nBest Parameters: {best_params}')
print(f'Best Silhouette Score: {best_score}')

# Step 6: Apply K-Medoids with the best parameters
kmedoids = KMedoids(n_clusters=best_params['n_clusters'], random_state=42)
kmedoids.fit(X_train_reduced)

df_train_split['Cluster'] = kmedoids.labels_
df_val_split['Cluster'] = kmedoids.predict(X_val_reduced)

# Step 7: Evaluate on validation set
if len(set(df_val_split['Cluster'])) > 1:
    sil_score_val = silhouette_score(X_val_reduced, df_val_split['Cluster'])
    ari_score_val = adjusted_rand_score(df_val_split['Topic'], df_val_split['Cluster'])
    print(f'Validation Silhouette Score: {sil_score_val}')
    print(f'Validation ARI: {ari_score_val}')

# Step 8: Apply clustering to test data
df_test['Cluster'] = kmedoids.predict(X_test_reduced)

if len(set(df_test['Cluster'])) > 1:
    sil_score_test = silhouette_score(X_test_reduced, df_test['Cluster'])
    ari_score_test = adjusted_rand_score(df_test['Topic'], df_test['Cluster'])
    print(f'Test Silhouette Score: {sil_score_test}')
    print(f'Test ARI: {ari_score_test}')