import redis
import json
import time
import pickle
import os
import logging
import numpy as np
import pandas as pd

from dotenv import load_dotenv

#Load enviroment variables
load_dotenv('.env')

# Set up logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

# Connect to Redis 
db = redis.Redis(
    host = os.getenv("REDIS_IP", "redis") #os.getenv('REDIS_IP')
    ,port = os.getenv('REDIS_PORT') 
    ,db = os.getenv('REDIS_DB_ID')
)


def load_pipeline(filename):
    """
    Load a trained machine learning model from a file.

    This function loads a pickled machine learning model from a given file.
    The model is expected to be a scikit-learn GridSearchCV object.

    Parameters:
    - filename (str): Path to the file containing the pickled model.

    Returns:
    - best_estimator (object): The best estimator (trained model) from GridSearchCV.
    - best_score (float): The best score achieved by the model during GridSearchCV.
    """
    if os.path.exists(filename):
        # Open the file in binary read mode
        with open(filename, "rb") as f:
            # Load the pickled model object from the file
            pipeline = pickle.load(f)
            # logger.info("model:", model_tmp)
    else:
        raise FileNotFoundError(f"The file {filename} does not exist.")

    return pipeline

def predict(data):
    """
    Make predictions using a trained machine learning model.

    This function takes input data, loads a pre-trained model, and makes predictions
    using the loaded model.

    Parameters:
    - data (list or numpy.ndarray): Input data for prediction.

    Returns:
    - model_name (str): Name of the loaded machine learning model.
    - class_name (int): Predicted class name for the input data.
    - pred_probability (float): Predicted probability of the predicted class.
    """
    

    # Define the path of the trained model file
    pipeline_file_path = "predict_model.pkl"

    # Load the pre-trained model and its best score
    pipeline = load_pipeline(pipeline_file_path)

    keys_order = ["FLAG_MASTERCARD", "FLAG_RESIDENCIAL_PHONE", "FLAG_PROFESSIONAL_PHONE", "SEX", "AGE", 
              "MONTHS_IN_RESIDENCE", "PAYMENT_DAY", "PROFESSION_CODE", "QUANT_BANKING_ACCOUNTS", 
              "QUANT_DEPENDANTS", "RESIDENCE_TYPE", "RESIDENCIAL_STATE", "STATE_OF_BIRTH", 
              "OCCUPATION_TYPE", "MARITAL_STATUS"]
    
    #["MONTHS_IN_RESIDENCE","PROFESSION_CODE","RESIDENCE_TYPE","OCCUPATION_TYPE"]
    
    df = pd.DataFrame([data],
                         columns=keys_order)
    
    float_columns = ["MONTHS_IN_RESIDENCE", "PROFESSION_CODE", "RESIDENCE_TYPE", "OCCUPATION_TYPE"]
    df[float_columns] = df[float_columns].astype(float)

    # Convert columns to integer
    int_columns = ["FLAG_MASTERCARD", "AGE", "PAYMENT_DAY", "QUANT_BANKING_ACCOUNTS", "QUANT_DEPENDANTS", "MARITAL_STATUS"]
    df[int_columns] = df[int_columns].astype(int)
    



    prediction = pipeline.predict(df)[0]  # Predicted class (0 or 1)
    probability = pipeline.predict_proba(df)[:, 0][0]  # Probability of class 0
    model_name = type(pipeline.named_steps['classifier']).__name__


    
    # Return the prediction results
    return model_name, prediction, probability

def classify_process():
    """
    Continuously process incoming jobs from the Redis queue.

    This function listens for incoming jobs in the Redis queue, runs the ML model on the data,
    stores the model prediction in Redis, and then waits for the next job.

    The loop runs indefinitely, continuously processing jobs from the queue.

    Note: This function should be run in a separate thread or process.

    Returns:
    - None
    """

    while True:
        # Take a new job from Redis
        msg_queue = db.brpop(os.getenv('REDIS_QUEUE'), float(os.getenv('SERVER_SLEEP')))

        if msg_queue is not None:
            # Extract the message content from the returned tuple
            msg_queue = msg_queue[1]

            # Run ML model on the given data
            newmsg = json.loads(msg_queue)


            model_name, class_name, pred_probability = predict(
                newmsg["data"]
            )

            # Store model prediction in a dictionary
            res_dict = {
                "model_name": model_name,
                "prediction": str(class_name),
                "score": round(np.float64(pred_probability), 2)
            }

            # Store the results on Redis using the original job ID as the key
            res_id = newmsg["id"]
            db.set(res_id, json.dumps(res_dict))

        # Sleep for a bit before processing the next job
        time.sleep(float(os.getenv('SERVER_SLEEP')))


if __name__ == "__main__":
    logger.info("....Starting ML....")
    classify_process()
