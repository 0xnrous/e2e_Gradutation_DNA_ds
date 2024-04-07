import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
# Import necessary libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_pickle("../data/processed/final_processed_data.pkl")

# Convert DNA sequence columns to string type
df['Parent_full_DNA_Seq'] = df['Parent_full_DNA_Seq'].astype(str)
df['Child_full_DNA_Seq'] = df['Child_full_DNA_Seq'].astype(str)

# Load vectorizer as kmer
vectorizer = joblib.load('../data/interim/kmer_model.sav')

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


# Truncate k-mer embeddings arrays
kmer_embeddings_c_truncated = kmer_embeddings_c[:len(X_train)]
kmer_embeddings_p_truncated = kmer_embeddings_p[:len(X_train)]


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
# Check the length of X_copy
        print("Length of X_copy:", len(X_copy))

        # Check the length of kmer_embeddings_c and kmer_embeddings_p
        print("Length of kmer_embeddings_c:", len(kmer_embeddings_c))
        print("Length of kmer_embeddings_p:", len(kmer_embeddings_p))

        # Check the consistency of train-test split
        print("Length of X_train:", len(X_train))
        print("Length of X_test:", len(X_test))
        print("Length of y_train:", len(y_train))
        print("Length of y_test:", len(y_test))

        
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
        X_copy = X.copy()
        for column in self.columns_to_drop:
            if column in X_copy.columns:
                X_copy.drop(column, axis=1, inplace=True)
        return X_copy
    
    

# Transform data using KmerEmbeddingTransformer
kmer_transformer = KmerEmbeddingTransformer(kmer_embeddings_c_truncated, kmer_embeddings_p_truncated)
X_train_transformed = kmer_transformer.fit_transform(X_train)
X_test_transformed = kmer_transformer.transform(X_test)

# Drop 'Parent_full_DNA_Seq' and 'Child_full_DNA_Seq' columns
X_train_transformed.drop(['Parent_full_DNA_Seq', 'Child_full_DNA_Seq'], axis=1, inplace=True)
X_test_transformed.drop(['Parent_full_DNA_Seq', 'Child_full_DNA_Seq'], axis=1, inplace=True)

# Drop specific columns
column_dropper = DropSpecificColumns(columns_to_drop=['child_gene_k_64', 'parent_gene_k_64'])
X_train_transformed = column_dropper.fit_transform(X_train_transformed)
X_test_transformed = column_dropper.transform(X_test_transformed)

# # Define the main pipeline with preprocessing steps
# pipeline = Pipeline([
#     ('scaler', MinMaxScaler()),
#     ('model', SVC(kernel='rbf', C=0.5, gamma='scale'))
# ])                                          
                                              
# Define the main pipeline with preprocessing steps
pipeline = Pipeline([
    # ('preprocessor', preprocessor),  # Preprocess DNA sequences
    #('kmer_embedding', KmerEmbeddingTransformer(kmer_embeddings_c, kmer_embeddings_p)),
    #('drop_specific_columns', DropSpecificColumns(columns_to_drop=['child_gene_k_64', 'parent_gene_k_64'])),
    ('scaler', MinMaxScaler()),
    ('model', XGBClassifier(
        learning_rate=0.3,
        n_estimators=15000,
        max_depth=15
    ))
])
# Fit the pipeline using transformed data
pipeline.fit(X_train_transformed, y_train)
# Save the pipeline
joblib.dump(pipeline, 'kmer_embedding_pipeline.pkl')

# # Load the pipeline
pipeline_loaded = joblib.load('kmer_embedding_pipeline.pkl')

# Evaluate the model
y_pred_test = pipeline_loaded.predict(X_test_transformed)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", accuracy_test)