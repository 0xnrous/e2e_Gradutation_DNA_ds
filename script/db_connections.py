# db connection
import pymongo
from pymongo import MongoClient
# Replace the placeholders with your MongoDB Atlas credentials and database name
def connect_to_mongodb():
        
    username = 'DinaAhmed'
    password = 'route123'
    cluster = 'cluster0.w8njcrd.mongodb.net'
    database_name = 'dna_test'

    # Construct the MongoDB Atlas connection string
    db_connection_uri = f'mongodb+srv://{username}:{password}@{cluster}/{database_name}?retryWrites=true&w=majority'

    # Create a MongoClient instance
    client = pymongo.MongoClient(db_connection_uri)

    # Access your database
    database = client[database_name]

    # Replace the placeholder with your MongoDB hostname and port
    hostname = 'cluster0.w8njcrd.mongodb.net'
    port = 27017  # Default MongoDB port

    # Create a MongoClient instance without authentication
    client = pymongo.MongoClient(hostname, port)

    # Access your database
    database = client['dna_test']
    collection = database['populations']
    cursor = collection.find({ }, 
                        {'name' : 1 ,
                        'phone' : 1 ,
                        'gender' : 1 ,
                        'birthdate' : 1 ,
                        'bloodType': 1 ,
                        'status' : 1 ,
                        'description': 1 , 
                        'createdAt' : 1 ,
                        'updatedAt' : 1 , 
                        'DNA_sequence': 1})


    name_list = []
    phone_list = []
    gender_list = []
    birthdate_list = []
    bloodType_list = []
    status_list = []
    description_list = []
    createdAt_list = []
    updatedAt_list = []
    DNA_sequence_list = []


    for row in cursor:
        name_list.append(row['name'])
        phone_list.append(row['phone'])
        gender_list.append(row['gender'])
        birthdate_list.append(row['birthdate'])
        bloodType_list.append(row['bloodType'])
        status_list.append(row['status'])
        description_list.append(row['description'])
        createdAt_list.append(row['createdAt'])
        updatedAt_list.append(row['updatedAt'])
        DNA_sequence_list.append(row['DNA_sequence'])

    # Create a dictionary from lists
    data = {
        'name': name_list,
        'phone': phone_list,
        'gender': gender_list,
        'birthdate': birthdate_list,
        'bloodType': bloodType_list,
        'status': status_list,
        'description': description_list,
        'createdAt': createdAt_list,
        'updatedAt': updatedAt_list,
        'DNA_sequence': DNA_sequence_list
    }

    # Create DataFrame from dictionary
    db_df = pd.DataFrame(data)

    # Close cursor and client connections
    cursor.close()
    client.close()

    # Return DataFrame
    return db_df