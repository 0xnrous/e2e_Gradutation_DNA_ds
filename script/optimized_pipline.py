import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Load the data
df = pd.read_pickle("../data/processed/final_processed_data.pkl")
df.drop(df.tail(100).index,inplace=True)

# Convert DNA sequence columns to string type
df['Parent_full_DNA_Seq'] = df['Parent_full_DNA_Seq'].astype(str)
df['Child_full_DNA_Seq'] = df['Child_full_DNA_Seq'].astype(str)

# Load precomputed k-mer embeddings for training data
kmer_embeddings_c = np.load('../data/interim/train_vectorizer_np/kmer_embeddings_c1.npy')
kmer_embeddings_p = np.load('../data/interim/train_vectorizer_np/kmer_embeddings_p1.npy')

# Select relevant columns
df_pre = df[['Parent_full_DNA_Seq', 'Child_full_DNA_Seq', 'target']]

# Split data into train and test sets
x = df_pre.drop('target', axis=1)
y = df_pre['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train = X_train[:len(kmer_embeddings_c)]
y_train = y_train[:len(kmer_embeddings_c)]

# Define transformer for k-mer embedding
class KmerEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, kmer_embeddings_c, kmer_embeddings_p):
        self.kmer_embeddings_c = kmer_embeddings_c
        self.kmer_embeddings_p = kmer_embeddings_p

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        max_length = min(len(X_copy), len(self.kmer_embeddings_c))
        X_copy = X_copy.iloc[:max_length]

        for i in range(len(self.kmer_embeddings_c[0])):
            X_copy[f'child_gene_k_{i}'] = [self.kmer_embeddings_c[j][i] for j in range(len(X_copy))]
            X_copy[f'parent_gene_k_{i}'] = [self.kmer_embeddings_p[j][i] for j in range(len(X_copy))]
        return X_copy

# Define custom transformer to drop specific columns
class DropSpecificColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1, errors='ignore').copy()

# Define the main pipeline with preprocessing and model training steps
#----------------------------------------------------------------------------------
# PIPELINE for XGB Classifier
#----------------------------------------------------------------------------------
# pipeline = Pipeline([
#     ('kmer_transformer', KmerEmbeddingTransformer(kmer_embeddings_c, kmer_embeddings_p)),
#     ('drop_columns', DropSpecificColumns(columns_to_drop=['Parent_full_DNA_Seq', 'Child_full_DNA_Seq'])),
#     ('scaler', MinMaxScaler()),
#     ('model', XGBClassifier(
#         learning_rate=0.3,
#         n_estimators=15000,
#         max_depth=15
#     ))
# ])

# Test Accuracy: 0.5411244365059114 / test 1 / XGBClassifier
# Time 34 min 10 sec
#----------------------------------------------------------------------------------
# PIPELINE for SVM Classifier
#----------------------------------------------------------------------------------
pipeline = Pipeline([
    ('kmer_transformer', KmerEmbeddingTransformer(kmer_embeddings_c, kmer_embeddings_p)),
    ('drop_columns', DropSpecificColumns(columns_to_drop=['Parent_full_DNA_Seq', 'Child_full_DNA_Seq'])),
    ('scaler', MinMaxScaler()),
    ('model', SVC(kernel='rbf', C=0.5, gamma='scale'))
    
])

# Test Accuracy: 0.6268273916388818 / test 2 / SVM
# Time 4 min 10 sec
#----------------------------------------------------------------------------------

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred_test = pipeline.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", accuracy_test)

# # Save the pipeline
# joblib.dump(pipeline, 'kmer_embedding_pipeline.pkl')
