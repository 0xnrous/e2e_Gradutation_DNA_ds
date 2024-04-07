import os
import logging
from logging.handlers import RotatingFileHandler
import json 
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pymongo
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Flask App Configuration
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('flask.log', maxBytes=204800, backupCount=20)
logger.addHandler(handler)

# Load ML models
loaded_model = joblib.load('../data/interim/kmer_model.sav')
with open('../models/model_xgb_clf_hyperparameter.pkl', 'rb') as f:
    model = joblib.load(f)

# MongoDB connection settings
# MONGODB_URI = os.getenv('mongodb+srv://{username}:{password}@{cluster}/{database_name}?retryWrites=true&w=majority')
# DB_NAME = os.getenv('dna_test')
# COLLECTION_NAME = os.getenv('populations')
# JSON Encoder Class 
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def connect_to_mongodb():
    username = 'DinaAhmed'
    password = 'route123'
    cluster = 'cluster0.w8njcrd.mongodb.net'
    database_name = 'dna_test'

    db_connection_uri = f'mongodb+srv://{username}:{password}@{cluster}/{database_name}?retryWrites=true&w=majority'
    
    client = pymongo.MongoClient(db_connection_uri)
    database = client[database_name]
    collection = database['populations']
    cursor = collection.find({}, {'_id': 0})  # Exclude _id field
    
    data = {key: [] for key in cursor[0].keys()}  # Initialize data dictionary
    
    for row in cursor:
        for key, value in row.items():
            data[key].append(value)
    
    db_df = pd.DataFrame(data)
    
    client.close()  # Close MongoDB connection
    return db_df

# Function to process uploaded files
def process_uploaded_file(path):
    files = os.listdir(path)
    data =  [] 
    for file in files :
        file_path = os.path.join(path , file)
        if os.path.isfile(file_path):
            with open(file_path , 'rb') as f:
                content = f.read() # read the (DNA_sequence) of the file from uploaded files
                data.append(content)
    return data

# Function to calculate likelihood ratio
def calculate_likelihood_ratio(father_allele, child_allele):
    """
    Calculate the likelihood ratio for paternity testing based on allele sequences.

    Parameters:
    - father_allele (str): Allele sequence of the father.
    - child_allele (str): Allele sequence of the child.

    Returns:
    - likelihood_ratio (float): Likelihood ratio indicating the probability of paternity.
    """
    match_count = 0
    total_count = 0
    length = len(father_allele) // 2

    allel1_father = father_allele[:length]
    allel2_father = father_allele[length:]

    allel1_child = child_allele[:length]
    allel2_child = child_allele[length:]

    # Compare allel1 sequences of father and child
    for i in range(len(allel1_father)):
        if allel1_father[i] == allel1_child[i]:
            match_count += 1
        total_count += 1

    # Compare allel2 sequences of father and child
    for i in range(len(allel2_father)):
        if allel2_father[i] == allel2_child[i]:
            match_count += 1
        total_count += 1

    likelihood_ratio = (match_count / total_count) * 100
    
    if likelihood_ratio > 77:
        return 1
    else :
        return 0

# Function to create DataFrame
def create_data_frame(child_seq, possible_parent_seq):
    repeated_value = child_seq[0]
    
    data = {
        'possiable_child': [repeated_value]* len(possible_parent_seq),
        'possiable_parent': possible_parent_seq
    }
    # convert the dictionary to a pandas DataFrame
    test_df = pd.DataFrame(data)
    return test_df



def toCategorical(df_):
    df_ca = df_.copy()
    df_ca['possiable_child'] = df_ca['possiable_child'].astype('category')
    df_ca['possiable_parent'] = df_ca['possiable_parent'].astype('category')
    return df_ca

# Function to encode DataFrame
def encode_df(df_):
    df_c = df_.copy()
    # kmer_embeddings_child_test = np.load('../data/interim/test_vectorizer_np/kmer_embeddings_child_test.npy')
    # kmer_embeddings_parent_test = np.load('../data/interim/test_vectorizer_np/kmer_embeddings_parent_test.npy')
    X_t_c = loaded_model.transform(df_c['possiable_child'])
    X_t_p = loaded_model.transform(df_c['possiable_parent'])
    
    kmer_embeddings_child_test = X_t_c.toarray()
    kmer_embeddings_parent_test = X_t_p.toarray()
    for i in range (0, len(kmer_embeddings_child_test[0])):
        df_c['child_gene_k_'+str(i)] = [kmer_embeddings_child_test[j][i] for j in range (0, len(df_c))]
        df_c['parent_gene_k_'+str(i)] = [kmer_embeddings_parent_test[j][i] for j in range (0, len(df_c))]
        
    return df_c.drop(['possiable_child' , 'possiable_parent'] , axis = 1)


# def encode_df(df_):
#     df_en = df_.copy()
#     X_t_c = loaded_model.transform(df_en['possiable_child'])
#     X_t_p = loaded_model.transform(df_en['possiable_parent'])

#     kmer_embeddings_child_test = X_t_c.toarray()
#     kmer_embeddings_parent_test = X_t_p.toarray()

#     # Create lists to hold column data
#     child_columns = []
#     parent_columns = []

#     # Populate column lists
#     for i in range(len(kmer_embeddings_child_test[0])):
#         child_columns.append([kmer_embeddings_child_test[j][i] for j in range(len(df_en))])
#         parent_columns.append([kmer_embeddings_parent_test[j][i] for j in range(len(df_en))])

#     # Concatenate column lists to DataFrame
#     df_en = pd.concat([df_en, pd.DataFrame(child_columns).add_prefix('child_gene_k_'), pd.DataFrame(parent_columns).add_prefix('parent_gene_k_')], axis=1)

#     # Drop original columns
#     df_en.drop(['possiable_child', 'possiable_parent'], axis=1, inplace=True)

#     return df_en

# Function to apply ML model
def apply_model(df_, df):
    if len(df.iloc[0,0 ]) != 78216:
        return [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
    y_pred = model.predict(df_)
    for i , v in enumerate(y_pred):
        if v == 1:
            actual_value = calculate_likelihood_ratio(df.iloc[i,1] , df.iloc[i,0])
            y_pred[i] = actual_value
    
    return y_pred

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/api', methods=['POST'])
def predict():
    file = request.files['file']
    file_content = file.read().decode('utf-8')
    file.close()
    child_seq = [file_content]
    a ,v , *_ = connect_to_mongodb()
    df = create_data_frame(child_seq , v)
    df_ca = toCategorical(df)
    df_encoded = encode_df(df_ca)
    y = apply_model(df_encoded , df)
    person_match = 'No Match in Database'
    for ind , val in enumerate(y):
        if val == 1:
            # person_match = a.iloc[ind]['name']
            person_match = {'_id': int(a.iloc[ind,0]), 'name': a.iloc[ind,1]}
            
    response =  jsonify({'person_match': person_match})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS , DELETE')
    response.headers.add('content-type', 'application/json')
    
    return response

if __name__ == '__main__':
    app.run(debug=True , port = 8000)
