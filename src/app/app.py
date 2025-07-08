import json
import logging
import os
import time
import uuid
from datetime import timedelta
from typing import Union


import database
import pandas as pd
import redis
import utils
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv

#Load enviroment variables
load_dotenv('.env')

# Set up logging
log_filename = "app_log.log"
logging.basicConfig(
    level=logging.ERROR,
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler = logging.FileHandler(filename=log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(file_handler)

# Current directory
current_dir = os.path.dirname(__file__)

# Connect to Redis
# Railway provides Redis URL, fallback to individual parameters for local development
redis_url = os.getenv('REDIS_URL') or os.getenv('REDIS_IP')
if redis_url and redis_url.startswith('redis://'):
    # Use URL-based connection for Railway
    db = redis.from_url(redis_url)
else:
    # Use individual parameters for local development
    db = redis.Redis(
        host = os.getenv('REDIS_IP', 'redis'),
        port = int(os.getenv('REDIS_PORT', 6379)),
        db = int(os.getenv('REDIS_DB_ID', 0))
    )

# Your FastAPI app
app = FastAPI()

# Load static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Endpoint Home page
@app.get("/", include_in_schema=False)
def home():
    return {"message": "Welcome to Credit Risk Analysis API!"}

# Health check endpoint for Railway
@app.get("/health", include_in_schema=False)
def health_check():
    """
    Health check endpoint for Railway deployment monitoring.
    
    Returns:
    - dict: Health status and service information
    """
    try:
        # Test Redis connection
        db.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "credit-risk-api",
        "redis": redis_status
    }

# Endpoint login
@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    """
    Display the login page template.

    Parameters:
    - request (Request): FastAPI request object.

    Returns:
    - TemplateResponse: The rendered login page template with the request context.
    """
    return templates.TemplateResponse("login.html", {"request": request})

# Endpoint token
@app.post("/token", response_class = HTMLResponse)
async def login_for_access_token(
    request: Request, username: str = Form(...), password: str = Form(...)
):
    """
    Authenticate user and generate access token.

    Parameters:
    - request (Request): FastAPI request object.
    - username (str): User's username.
    - password (str): User's password.

    Returns:
    - HTMLResponse: Login page with error message if login fails, otherwise index.html content.
    """

    user = utils.authenticate_user(database.fake_users_db, username, password)

    if not user:
        error_message = "Incorrect username or password"
        # Return the login page with an error message
        return templates.TemplateResponse(
            "login.html", {"request": request, "error_message": error_message}
        )

    access_token_expires = timedelta(minutes = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'))) #settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = utils.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    # Render the index.html page directly in the response
    index_page_content = templates.get_template("index.html").render(request=request)
    response = HTMLResponse(content = index_page_content)

    # Set the token as a cookie
    response.set_cookie("access_token", access_token)
    return response

# Endpoint index. Render the loan prediction form using Jinja2 template
@app.get("/index/", response_class=HTMLResponse, include_in_schema=False)
async def get_loan_prediction_form(request: Request):
    # Render the template with the necessary context
    return templates.TemplateResponse("index.html", {"request": request})





@app.post(
    "/prediction", 
    response_class=HTMLResponse,
    summary="Predict loan approval",
    description="Process loan application data and predict whether the loan will be approved or rejected.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/x-www-form-urlencoded": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "NAME": {"type": "string", "example": "John Doe", "description": "Customer's full name"},
                            "AGE": {"type": "integer", "example": 49, "description": "Applicant's age at the moment of submission"},
                            "SEX": {"type": "string", "example": "F", "description": "Gender (M=Male, F=Female)"},
                            "MARITAL_STATUS": {"type": "integer", "example": 2, "description": "Marital status (encoding not informed)"},
                            "PAYMENT_DAY": {"type": "integer", "example": 20, "description": "Day of the month for bill payment, chosen by the applicant (1,5,10,15,20,25)"},
                            "RESIDENCIAL_STATE": {"type": "string", "example": "CE", "description": "State of residence"},
                            "MONTHS_IN_RESIDENCE": {"type": "float", "example": 18.0, "description": "Time in the current residence in months"},
                            "PROFESSION_CODE": {"type": "float", "example": 2.0, "description": "Applicant's profession code (encoding not informed)"},
                            "QUANT_BANKING_ACCOUNTS": {"type": "integer", "example": 0, "description": "Number of banking accounts (0,1,2)"},
                            "QUANT_DEPENDANTS": {"type": "integer", "example": 1, "description": "Number of dependants (0, 1, 2, ...)"},
                            "RESIDENCE_TYPE": {"type": "float", "example": 1.0, "description": "Type of residence (encoding not informed: owned, mortgage, rented, parents, family, etc.)"},
                            "STATE_OF_BIRTH": {"type": "string", "example": "CE", "description": "State of birth (Brazilian states, XX, missing)"},
                            "OCCUPATION_TYPE": {"type": "float", "example": 4.0, "description": "Occupation type (encoding not informed)"},
                            "FLAG_MASTERCARD": {"type": "integer", "example": 1, "description": "Flag indicating if the applicant is a MASTERCARD credit card holder (0.1)"},
                            "FLAG_RESIDENCIAL_PHONE": {"type": "string", "example": "N", "description": "Indicates if the applicant possesses a home phone (Y,N)"},
                            "FLAG_PROFESSIONAL_PHONE": {"type": "string", "example": "Y", "description": "Indicates if the professional phone number was supplied (Y,N)"}
                        }
                    }
                }
            }
        }
    }
)
async def predict(
    request: Request,
    NAME: str = Form(..., description="Customer's full name"),
    AGE: int = Form(..., description="Applicant's age at the moment of submission"),
    SEX: str = Form(..., description="Gender (M=Male, F=Female)"),
    MARITAL_STATUS: int = Form(..., description="Marital status (encoding not informed)"),
    PAYMENT_DAY: int = Form(..., description="Day of the month for bill payment, chosen by the applicant (1,5,10,15,20,25)"),
    RESIDENCIAL_STATE: str = Form(..., description="State of residence"),
    MONTHS_IN_RESIDENCE: float = Form(..., description="Time in the current residence in months"),
    PROFESSION_CODE: float = Form(..., description="Applicant's profession code (encoding not informed)"),
    QUANT_BANKING_ACCOUNTS: int = Form(..., description="Number of banking accounts (0,1,2)"),
    QUANT_DEPENDANTS: int = Form(..., description="Number of dependants (0, 1, 2, ...)"),
    RESIDENCE_TYPE: float = Form(..., description="Type of residence (encoding not informed: owned, mortgage, rented, parents, family, etc.)"),
    STATE_OF_BIRTH: str = Form(..., description="State of birth (Brazilian states, XX, missing)"),
    OCCUPATION_TYPE: float = Form(..., description="Occupation type (encoding not informed)"),
    FLAG_MASTERCARD: int = Form(..., description="Flag indicating if the applicant is a MASTERCARD credit card holder (0.1)"),
    FLAG_RESIDENCIAL_PHONE: str = Form(..., description="Indicates if the applicant possesses a home phone (Y,N)"),
    FLAG_PROFESSIONAL_PHONE: str = Form(..., description="Indicates if the professional phone number was supplied (Y,N)"),
):
    """
    Handle user's credit prediction request using a machine learning model.

    Parameters:
    - request (Request): FastAPI request object.
    - NAME (str): User's name.
    - AGE (int): Applicant's age at the moment of submission.
    - SEX (str): Gender (M=Male, F=Female).
    - MARITAL_STATUS (int): Marital status (encoding not informed).
    - PAYMENT_DAY (int): Day of the month for bill payment, chosen by the applicant (1,5,10,15,20,25).
    - RESIDENCIAL_STATE (str): State of residence.
    - MONTHS_IN_RESIDENCE (float): Time in the current residence in months.
    - PROFESSION_CODE (float): Applicant's profession code (encoding not informed).
    - QUANT_BANKING_ACCOUNTS (int): Number of banking accounts (0,1,2).
    - QUANT_DEPENDANTS (int): Number of dependants (0, 1, 2, ...).
    - RESIDENCE_TYPE (float): Type of residence (encoding not informed: owned, mortgage, rented, parents, family, etc.).
    - STATE_OF_BIRTH (str): State of birth (Brazilian states, XX, missing).
    - OCCUPATION_TYPE (float): Occupation type (encoding not informed).
    - FLAG_MASTERCARD (int): Flag indicating if the applicant is a MASTERCARD credit card holder (0.1).
    - FLAG_RESIDENCIAL_PHONE (str): Indicates if the applicant possesses a home phone (Y,N).
    - FLAG_PROFESSIONAL_PHONE (str): Indicates if the professional phone number was supplied (Y,N).

    Returns:
    - TemplateResponse: FastAPI template response containing prediction outcome.
    """

    form_data = await request.form()  # Get all form data as a Starlette FormData object
    data = dict(form_data)  # Convert it into a dictionary

    remaining = data.copy()

    user_name = remaining.pop('NAME')

    keys_order = ["FLAG_MASTERCARD", "FLAG_RESIDENCIAL_PHONE", "FLAG_PROFESSIONAL_PHONE", "SEX", "AGE", 
              "MONTHS_IN_RESIDENCE", "PAYMENT_DAY", "PROFESSION_CODE", "QUANT_BANKING_ACCOUNTS", 
              "QUANT_DEPENDANTS", "RESIDENCE_TYPE", "RESIDENCIAL_STATE", "STATE_OF_BIRTH", 
              "OCCUPATION_TYPE", "MARITAL_STATUS"]

    ordered_values = [remaining[key] for key in keys_order]

    

    # Generate an id for the classification then
    data_message = {"id": str(uuid.uuid4()), "data": ordered_values}

    job_data = json.dumps(data_message)
    job_id = data_message["id"]

    # Send the job to the model service using Redis
    db.lpush(os.getenv("REDIS_QUEUE"), job_data)

    # Wait for result model
    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        output = db.get(job_id)

        if output is not None:
            # Process the result and extract prediction and score
            output = json.loads(output.decode("utf-8"))
            model_name = output["model_name"]
            prediction = output["prediction"]
            score = output["score"]

            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(float(os.getenv('API_SLEEP')))

    # Determine the output message
    if int(prediction) == 1:
        prediction = "Dear {name}, your loan is rejected!\n with a score of {proba:.0f}".format(
            name=user_name, proba=score * 100
        )
    else:
        prediction = "Dear {name}, your loan is approved!\n with a score of {proba:.0f}".format(
            name = user_name, proba=score * 100
        )

    context = {
        "request": request,
        "model_name": model_name,
        "result": prediction,
    }

    # Render the prediction response using Jinja2 templates and return it
    return templates.TemplateResponse("prediction.html", context)
