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
db = redis.Redis(
    host = os.getenv('REDIS_IP', 'redis') #os.getenv('REDIS_IP')
    ,port = os.getenv('REDIS_PORT') 
    ,db = os.getenv('REDIS_DB_ID')
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

# Endpoint Prediction page
# @app.post(
#     "/prediction", 
#     response_class=HTMLResponse,
#     summary="Predict loan approval",
#     description="Process loan application data and predict whether the loan will be approved or rejected.",
#     openapi_extra={
#         "requestBody": {
#             "content": {
#                 "application/x-www-form-urlencoded": {
#                     "schema": {
#                         "type": "object",
#                         "properties": {
#                             "name": {"type": "string", "example": "John Doe", "description": "Customer's full name"},
#                             "age": {"type": "string", "example": "26-35", "description": "Age range of the customer (e.g., 26-35, 36-45, 46-60, <18, >60)"},
#                             "sex": {"type": "integer", "example": 1, "description": "Gender (Male=1, Female=0)"},
#                             "marital_status": {"type": "string", "example": "single", "description": "Marital status (other, single)"},
#                             "monthly_income_tot": {"type": "string", "example": "1320-3323", "description": "Monthly income range (1320-3323, 3323-8560, 650-1320, >8560)"},
#                             "payment_day": {"type": "integer", "example": 1, "description": "Payment day (0 = 1-14, 1 = 15-30)"},
#                             "residential_state": {"type": "string", "example": "NY", "description": "State of residence abbreviation"},
#                             "months_in_residence": {"type": "string", "example": ">12", "description": "Months in residence (6-12, >12)"},
#                             "product": {"type": "string", "example": "2", "description": "Product type (2, 7)"},
#                             "flag_company": {"type": "string", "example": "on", "description": "Has company association (on/off)"},
#                             "flag_dependants": {"type": "string", "example": "on", "description": "Has dependants (on/off)"},
#                             "quant_dependants": {"type": "integer", "example": 1, "description": "Number of dependants (1, 2, 3)"},
#                             "flag_residencial_phone": {"type": "string", "example": "on", "description": "Has residential phone (on/off)"},
#                             "flag_professional_phone": {"type": "string", "example": "on", "description": "Has professional phone (on/off)"},
#                             "flag_email": {"type": "string", "example": "on", "description": "Has email (on/off)"},
#                             "flag_cards": {"type": "string", "example": "on", "description": "Has cards (on/off)"},
#                             "flag_residence": {"type": "string", "example": "on", "description": "Has residence (on/off)"},
#                             "flag_banking_accounts": {"type": "string", "example": "on", "description": "Has banking accounts (on/off)"},
#                             "flag_personal_assets": {"type": "string", "example": "on", "description": "Has personal assets (on/off)"},
#                             "flag_cars": {"type": "string", "example": "on", "description": "Has cars (on/off)"}
#                         }
#                     }
#                 }
#             }
#         }
#     }
# )
# def predict(
#     request: Request,
#     name: str = Form(..., description="Customer's full name"),
#     age: str = Form(..., description="Age range of the customer (e.g., 26-35, 36-45, 46-60, <18, >60)"),
#     sex: int = Form(..., description="Gender (Male=1, Female=0)"),
#     marital_status: str = Form(..., description="Marital status (other, single)"),
#     monthly_income_tot: str = Form(..., description="Monthly income range (1320-3323, 3323-8560, 650-1320, >8560)"),
#     payment_day: int = Form(..., description="Payment day (0 = 1-14, 1 = 15-30)"),
#     residential_state: str = Form(..., description="State of residence abbreviation"),
#     months_in_residence: Union[str, None] = Form(None, description="Months in residence (6-12, >12)"),
#     product: str = Form(..., description="Product type (2, 7)"),
#     flag_company: Union[str, None] = Form(None, description="Has company association (on/off)"),
#     flag_dependants: Union[str, None] = Form(None, description="Has dependants (on/off)"),
#     quant_dependants: int = Form(..., description="Number of dependants (1, 2, 3)"),
#     flag_residencial_phone: Union[str, None] = Form(None, description="Has residential phone (on/off)"),
#     flag_professional_phone: Union[str, None] = Form(None, description="Has professional phone (on/off)"),
#     flag_email: Union[str, None] = Form(None, description="Has email (on/off)"),
#     flag_cards: Union[str, None] = Form(None, description="Has cards (on/off)"),
#     flag_residence: Union[str, None] = Form(None, description="Has residence (on/off)"),
#     flag_banking_accounts: Union[str, None] = Form(None, description="Has banking accounts (on/off)"),
#     flag_personal_assets: Union[str, None] = Form(None, description="Has personal assets (on/off)"),
#     flag_cars: Union[str, None] = Form(None, description="Has cars (on/off)"),
# ):
#     """
#     Handle user's credit prediction request using a machine learning model.

#     Parameters:
#     - request (Request): FastAPI request object.
#     - name (str): User's name.
#     - age (str): Age range.
#     - sex (int): Gender (Male=1, Female=0).
#     - marital_status (str): Marital status (other, single).
#     - monthly_income_tot (str): Monthly income range.
#     - payment_day (int): Payment day (0 = 1-14, 1 = 15-30).
#     - residential_state (str): State of residence.
#     - months_in_residence (Union[str, None]): Months in residence.
#     - product (str): Product type.
#     - flag_company (Union[str, None]): Flag for company.
#     - flag_dependants (Union[str, None]): Flag for dependants.
#     - quant_dependants (int): Number of dependants.
#     - flag_residencial_phone (Union[str, None]): Flag for residential phone.
#     - flag_professional_phone (Union[str, None]): Flag for professional phone.
#     - flag_email (Union[str, None]): Flag for email.
#     - flag_cards (Union[str, None]): Flag for cards.
#     - flag_residence (Union[str, None]): Flag for residence.
#     - flag_banking_accounts (Union[str, None]): Flag for banking accounts.
#     - flag_personal_assets (Union[str, None]): Flag for personal assets.
#     - flag_cars (Union[str, None]): Flag for cars.

#     Returns:
#     - TemplateResponse: FastAPI template response containing prediction outcome.
#     """

#     # Rest of your function stays the same
#     # Load template of JSON file containing columns name
#     schema_name = "columns_set.json"

#     # Directory where the schema is stored
#     schema_dir = os.path.join(current_dir, schema_name)
#     with open(schema_dir, "r") as f:
#         cols = json.loads(f.read())
#     schema_cols = cols["data_columns"]

#     # Parse the Categorical columns (Greater than one column)
#     # RESIDENCIAL_STATE_ (AL, ...)
#     try:
#         col = "RESIDENCIAL_STATE_" + str(residential_state)
#         if col in schema_cols.keys():
#             schema_cols[col] = 1
#         else:
#             schema_cols[col] = 0
#     except:
#         pass

#     # MARITAL_STATUS_ (other, single)
#     try:
#         col = "MARITAL_STATUS_" + str(marital_status)
#         if col in schema_cols.keys():
#             schema_cols[col] = 1
#         else:
#             schema_cols[col] = 0
#     except:
#         pass

#     # MONTHLY_INCOMES_TOT_ (1320-3323, 3323-8560, 650-1320, >8560)
#     try:
#         col = "MONTHLY_INCOMES_TOT_" + str(monthly_income_tot)
#         if col in schema_cols.keys():
#             schema_cols[col] = 1
#         else:
#             schema_cols[col] = 0
#     except:
#         pass

#     # QUANT_DEPENDANTS_ (1, 2, 3)
#     try:
#         col = "QUANT_DEPENDANTS_" + str(quant_dependants)
#         if col in schema_cols.keys():
#             schema_cols[col] = 1
#         else:
#             schema_cols[col] = 0
#     except:
#         pass

#     # MONTHS_IN_RESIDENCE_ (6-12, >12)
#     try:
#         col = "MONTHS_IN_RESIDENCE_" + str(months_in_residence)
#         if col in schema_cols.keys():
#             schema_cols[col] = 1
#         else:
#             schema_cols[col] = 0
#     except:
#         pass

#     # PRODUCT_ (2, 7)
#     try:
#         col = "PRODUCT_" + str(product)
#         if col in schema_cols.keys():
#             schema_cols[col] = 1
#         else:
#             schema_cols[col] = 0
#     except:
#         pass

#     # AGE_ (26-35, 36-45, 46-60, <18, >60)
#     try:
#         col = "AGE_" + str(age)
#         if col in schema_cols.keys():
#             schema_cols[col] = 1
#         else:
#             schema_cols[col] = 0
#     except:
#         pass

#     flag_company = True if flag_company in ["on", "1"] else False
#     flag_dependants = True if flag_dependants in ["on", "1"] else False
#     flag_residencial_phone = True if flag_residencial_phone in ["on", "1"] else False
#     flag_professional_phone = True if flag_professional_phone in ["on", "1"] else False
#     flag_email = True if flag_email in ["on", "1"] else False
#     flag_cards = True if flag_cards in ["on", "1"] else False
#     flag_residence = True if flag_residence in ["on", "1"] else False
#     flag_banking_accounts = True if flag_banking_accounts in ["on", "1"] else False
#     flag_personal_assets = True if flag_personal_assets in ["on", "1"] else False
#     flag_cars = True if flag_cars in ["on", "1"] else False

#     # Parse the Numerical columns (One column)
#     schema_cols["PAYMENT_DAY_15-30"] = payment_day
#     schema_cols["FLAG_RESIDENCIAL_PHONE_Y"] = flag_residencial_phone
#     schema_cols["FLAG_PROFESSIONAL_PHONE_Y"] = flag_professional_phone
#     schema_cols["COMPANY_Y"] = flag_company
#     schema_cols["FLAG_EMAIL_1"] = flag_email
#     schema_cols["SEX_M"] = sex
#     schema_cols["HAS_DEPENDANTS_True"] = flag_dependants
#     schema_cols["HAS_RESIDENCE_True"] = flag_residence
#     schema_cols["HAS_CARDS_True"] = flag_cards
#     schema_cols["HAS_BANKING_ACCOUNTS_True"] = flag_banking_accounts
#     schema_cols["HAS_PERSONAL_ASSETS_True"] = flag_personal_assets
#     schema_cols["HAS_CARS_True"] = flag_cars

#     # Convert the JSON into data frame
#     df = pd.DataFrame(data={k: [v] for k, v in schema_cols.items()}, dtype=float)
#     # Replace NaN values with 0.0
#     df = df.fillna(0.0)


    
#     # Debug the dimensions
#     print(f"Generated dataframe shape: {df.shape}")
#     with open("debug_features.json", "w") as f:
#         json.dump({"features": df.columns.tolist(), "count": df.shape[1]}, f, indent=2)

#     # CRITICAL FIX: Ensure EXACTLY 57 features are passed to the model
#     if df.shape[1] != 57:
#         print(f"Feature count mismatch: Have {df.shape[1]}, need 57")
        
#         # The key issue: need to ensure consistent columns for the model
#         # If we have more than 57 features, trim the data down to exactly 57
#         if df.shape[1] > 57:
#             # Sort out features by importance (1's first, then alphabetical)
#             # This prioritizes features with actual values
#             col_order = df.iloc[0].sort_values(ascending=False).index.tolist()
            
#             # Keep only the 57 most important features
#             df = df[col_order[:57]]
#             print(f"Kept top 57 features, new shape: {df.shape}")
#         else:
#             # If we have fewer than 57 features, pad with zeros
#             missing = 57 - df.shape[1]
#             for i in range(missing):
#                 df[f"padding_{i}"] = 0
#             print(f"Padded features to 57, new shape: {df.shape}")

#         # Final check
#         if df.shape[1] != 57:
#             raise ValueError(f"Failed to adjust feature count to 57 (current: {df.shape[1]})")

#     # Write the final columns for reference
#     with open("final_features.json", "w") as f:
#         json.dump({
#             "final_features": df.columns.tolist(), 
#             "count": df.shape[1]
#         }, f, indent=2)






#     # Generate an id for the classification then
#     data_message = {"id": str(uuid.uuid4()), "data": df.iloc[0].values.tolist()}

#     job_data = json.dumps(data_message)
#     job_id = data_message["id"]

#     # Send the job to the model service using Redis
#     db.lpush(os.getenv("REDIS_QUEUE"), job_data)

#     # Wait for result model
#     # Loop until we received the response from our ML model
#     while True:
#         # Attempt to get model predictions using job_id
#         output = db.get(job_id)

#         if output is not None:
#             # Process the result and extract prediction and score
#             output = json.loads(output.decode("utf-8"))
#             model_name = output["model_name"]
#             model_score = output["model_score"]
#             prediction = output["prediction"]
#             score = output["score"]

#             db.delete(job_id)
#             break

#         # Sleep some time waiting for model results
#         time.sleep(float(os.getenv('API_SLEEP')))

#     # Determine the output message
#     if int(prediction) == 1:
#         prediction = "Dear {name}, your loan is rejected!\n with a probability of {proba}".format(
#             name=name, proba=score
#         )
#     else:
#         prediction = "Dear {name}, your loan is approved!".format(name = name)

#     context = {
#         "request": request,
#         "model_name": model_name,
#         "model_score": model_score,
#         "result": prediction,
#     }

#     # Render the prediction response using Jinja2 templates and return it
#     return templates.TemplateResponse("prediction.html", context)




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
                            "name": {"type": "string", "example": "John Doe", "description": "Customer's full name"},
                            "age": {"type": "integer", "example": 49, "description": "Applicant's age at the moment of submission"},
                            "sex": {"type": "string", "example": "F", "description": "Gender (M=Male, F=Female)"},
                            "marital_status": {"type": "integer", "example": 2, "description": "Marital status (encoding not informed)"},
                            "payment_day": {"type": "integer", "example": 20, "description": "Day of the month for bill payment, chosen by the applicant (1,5,10,15,20,25)"},
                            "residential_state": {"type": "string", "example": "CE", "description": "State of residence"},
                            "months_in_residence": {"type": "float", "example": 18.0, "description": "Time in the current residence in months"},
                            "profession_code": {"type": "float", "example": 2.0, "description": "Applicant's profession code (encoding not informed)"},
                            "quant_banking_accounts": {"type": "integer", "example": 0, "description": "Number of banking accounts (0,1,2)"},
                            "quant_dependants": {"type": "integer", "example": 1, "description": "Number of dependants (0, 1, 2, ...)"},
                            "residence_type": {"type": "float", "example": 1.0, "description": "Type of residence (encoding not informed: owned, mortgage, rented, parents, family, etc.)"},
                            "state_of_birth": {"type": "string", "example": "CE", "description": "State of birth (Brazilian states, XX, missing)"},
                            "occupation_type": {"type": "float", "example": 4.0, "description": "Occupation type (encoding not informed)"},
                            "flag_mastercard": {"type": "integer", "example": 1, "description": "Flag indicating if the applicant is a MASTERCARD credit card holder (0.1)"},
                            "flag_residencial_phone": {"type": "string", "example": "N", "description": "Indicates if the applicant possesses a home phone (Y,N)"},
                            "flag_professional_phone": {"type": "string", "example": "Y", "description": "Indicates if the professional phone number was supplied (Y,N)"}
                        }
                    }
                }
            }
        }
    }
)
async def predict(
    request: Request,
    name: str = Form(..., description="Customer's full name"),
    age: int = Form(..., description="Applicant's age at the moment of submission"),
    sex: str = Form(..., description="Gender (M=Male, F=Female)"),
    marital_status: int = Form(..., description="Marital status (encoding not informed)"),
    payment_day: int = Form(..., description="Day of the month for bill payment, chosen by the applicant (1,5,10,15,20,25)"),
    residential_state: str = Form(..., description="State of residence"),
    months_in_residence: float = Form(..., description="Time in the current residence in months"),
    profession_code: float = Form(..., description="Applicant's profession code (encoding not informed)"),
    quant_banking_accounts: int = Form(..., description="Number of banking accounts (0,1,2)"),
    quant_dependants: int = Form(..., description="Number of dependants (0, 1, 2, ...)"),
    residence_type: float = Form(..., description="Type of residence (encoding not informed: owned, mortgage, rented, parents, family, etc.)"),
    state_of_birth: str = Form(..., description="State of birth (Brazilian states, XX, missing)"),
    occupation_type: float = Form(..., description="Occupation type (encoding not informed)"),
    flag_mastercard: int = Form(..., description="Flag indicating if the applicant is a MASTERCARD credit card holder (0.1)"),
    flag_residencial_phone: str = Form(..., description="Indicates if the applicant possesses a home phone (Y,N)"),
    flag_professional_phone: str = Form(..., description="Indicates if the professional phone number was supplied (Y,N)"),
):
    """
    Handle user's credit prediction request using a machine learning model.

    Parameters:
    - request (Request): FastAPI request object.
    - name (str): User's name.
    - age (int): Applicant's age at the moment of submission.
    - sex (str): Gender (M=Male, F=Female).
    - marital_status (int): Marital status (encoding not informed).
    - payment_day (int): Day of the month for bill payment, chosen by the applicant (1,5,10,15,20,25).
    - residential_state (str): State of residence.
    - months_in_residence (float): Time in the current residence in months.
    - profession_code (float): Applicant's profession code (encoding not informed).
    - quant_banking_accounts (int): Number of banking accounts (0,1,2).
    - quant_dependants (int): Number of dependants (0, 1, 2, ...).
    - residence_type (float): Type of residence (encoding not informed: owned, mortgage, rented, parents, family, etc.).
    - state_of_birth (str): State of birth (Brazilian states, XX, missing).
    - occupation_type (float): Occupation type (encoding not informed).
    - flag_mastercard (int): Flag indicating if the applicant is a MASTERCARD credit card holder (0.1).
    - flag_residencial_phone (str): Indicates if the applicant possesses a home phone (Y,N).
    - flag_professional_phone (str): Indicates if the professional phone number was supplied (Y,N).

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
        prediction = "Dear {name}, your loan is rejected!\n with a probability of {proba}".format(
            name=user_name, proba=score
        )
    else:
        prediction = "Dear {name}, your loan is approved!".format(name = user_name)

    context = {
        "request": request,
        "model_name": model_name,
        "result": prediction,
    }

    # Render the prediction response using Jinja2 templates and return it
    return templates.TemplateResponse("prediction.html", context)