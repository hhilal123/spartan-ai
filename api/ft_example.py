from langchain.utilities import SQLDatabase
from langchain.embeddings.openai import OpenAIEmbeddings
from agent.chatprompt import (
    SQL_QUERY_DESCPTION,
    SQL_PREFIX,
    # SCHEMA,
    SDCASE_EXT,
    ASSETS_EXT,
    FEW_SHOT_EXAMPLE_TEMPLATE,
    FINAL_FEWSHOT_ASSET_PROMPT_TEMPLATE,
    FINAL_FEWSHOT_ASSET_PROMPT_TEMPLATE_001,
    ASSETS_EXT_001
)

import datetime
from datetime import date
import asyncio
from difflib import IS_CHARACTER_JUNK, IS_LINE_JUNK
import os
import re
import time

import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, jsonify, request, send_file
from langchain.schema import OutputParserException
from langchain.chat_models import ChatOpenAI
# from langchain.chains import SQLDatabaseSequentialChain
import langchain
# from langchain.chains.sql_database.prompt import _DEFAULT_TEMPLATE, PROMPT_SUFFIX
# from langchain.prompts.prompt import PromptTemplate
import openai.error as aierror
from flask_cors import CORS


from agent.callbacks import LoggingCallbackHander, Grabber
from agent.chatsqlagent import create_sql_agent
from agent.sql import SQLDatabaseExt
import agent.database as memory
from dotenv import load_dotenv
import pinecone
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agent.tools import QueryUserNames
from agent.util import CustomError, UtilFunctions
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
util = UtilFunctions()


# nltk.data.path.append("E:\\OpenAI.API\\.venv\\Lib\\nltk_data")
nltk.download('punkt')
load_dotenv('.env')  # Added for PROD IIS Deployment - NS
# NOTE: Change user memory to use an SQLite database

app = Flask("VComGPTAPI")

# Changes: Nagasai
# Date: 2023-12-14
# Open AI integration - PROD Version
# SD Cases - v1 with gpt3 trubo
# Assets - v2 with fine tune model
# v3 for Assets with vector database & Azure AI with GPT 4 (not optimal, can be improved by generalizing the code)
# v4 for Assets with vector database & Open AI with GPT 4 (not optimal, can be improved by generalizing the code)

# Define the allowed origins and localhost (replace with your specific URLs)
allowed_origins = [
    "https://ipathdev.vcomsolutions.com",
    "https://ipathuat.vcomsolutions.com",
    "https://ipath.vcomsolutions.com",
    "https://vmannagerdev.vcomsolutions.com",
    "https://vmanageruat.vcomsolutions.com",
    "https://vmanager.vcomsolutions.com",
    "https://10.0.3.15:8082/",  # hosted second ip
    "http://localhost:5000",   # Example for localhost
]

# CORS(app, supports_credentials=True, expose_headers='Authorization')
# CORS(app, resources={r"/api/*": {"origins": "*"}})
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Normalized module name mapped to a view and static-value columns
# NOTE: Deprecated in /api/v2
viewselector = {
    "softwarecases": [
        "AI_sdCase_GetAll",
        [
            "Severity",
            "Application",
            "Status",
            "CaseType",
            "Priority",
            "Module",
            "ResolutionReason"
        ]
    ],
    "thing": ["AI_Asset_Info"]
}

# Compile the regex for data extraction
image_regex = re.compile(r"!\[[\w ]*\]\((.+)\)\n*")

# Grab view names from the viewselector dict
views = [col[0] for col in viewselector.values()]

server = os.environ.get("SQL_SERVER")
database = os.environ.get("SQL_DATABASE")
username = os.environ.get("SQL_USERNAME")
password = os.environ.get("SQL_PASSWORD")
driver = os.environ.get("SQL_DRIVER")
model = os.environ.get("OPENAI_MODEL")
model_ft = os.environ.get("OPENAI_FT_MODEL")
environment = os.environ.get("ENVIRONMENT", "")
memory_len = int(os.environ.get("MEMORY_LENGTH", "3"))  # Default to 5
# Set langchain debug.
langchain.debug = os.environ.get("VComGPTDEBUG", "").lower() == "true"

# setting tempature gloablly to fit all api calls
temperature = float(os.environ.get("TEMPERATURE", "0.2"))

model_gpt4 = os.environ.get("OPENAI_GPT4_MODEL")

# Check for empty values
for item in (server, database, username, password, driver, model, model_ft, model_gpt4):
    if not item:
        raise ValueError("Missing an envirornment variable.")

sqldb = SQLDatabaseExt.from_uri(
    f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}",
    sample_rows_in_table_info=0,
    max_string_length=8000,  # For notes table
    include_tables=views,  # Removed for /api/v2
    view_support=True
)

# In-memory database for AI user memory
asyncio.run(memory.start_engine())

# Initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)
index_name = "vchat"


@app.get("/blob/image/<image_id>")
def return_image(image_id):
    """Send an image if exists"""
    image_file = os.path.join("static", image_id + ".png")
    if os.path.exists(image_file):
        return send_file(image_file, mimetype="image/png")
    return {"error": "Image does not exist"}, 404




@app.route("/api/v4/getPyResponse", methods=["GET", "POST", "OPTIONS"])
async def v4_return_response():
    """Handle preflight and post requests to get a response from the AI"""

    # Setting AI Model
    AIModel = model_gpt4

    embeddings = OpenAIEmbeddings()

    temperature = float(os.environ.get("TEMPERATURE", "0.2"))

    views = ["AI_Asset_Info", "AI_InventoryAttribute"]

    # print(embeddings)

    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response_headers = {
            "Access-Control-Allow-Origin": request.headers.get("Origin"),
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Methods": "POST, OPTIONS",  # Allow necessary methods
        }
        return '', 200, response_headers

    if request.is_json:
        data = request.get_json()  # If the request contains JSON data
    else:
        data = request.form  # If the request contains form data (e.g., from an HTML form)

    try:
        user_id = int(data.get("UserId"))
        customer_id = int(data.get("CustomerID"))
    except TypeError:
        return {"error": "Invalid request, could not find valid integer 'UserId' or 'CustomerID'"}

    module = data.get("Module")
    norm_module = util.normalize_string(module)

    promptName = data.get("PromptName").strip()
    question = data.get("Request").strip()
    referer = request.headers.get("Origin")
    referer = request.headers.get("Origin")

    if (referer is None):
        internal = 'false'
    elif (referer.find("localhost") == -1):
        internal = 'false'
    else:
        internal = 'true'

    internal = bool(internal)
    views = data.get("TableorViews").split(",")
    # cols = data.get("ColumnLookup").split("|")
    # internal = bool(data.get("IsInternal"))
    # allcols = [group.split(",") for group in cols]

    # Setting the "Content-Type" header in the response
    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": request.headers.get("Origin"),
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }

    # Get user memory from db
    mem = await memory.get_user_memory(user_id, memory_len)

    # Define response params and create an agent
    res_message = "Success"
    error_code = 200
    graph_image_url = None
    ai_response = ""
    sql_query = ""

    # grabber = Grabber()

    query = "Question: " + question

    from langchain.vectorstores import Pinecone

    # Initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    index_name = "vchat"
    # if you already have an index, you can load it like this
    index = pinecone.Index("vchat")
    vectorstore = Pinecone(index, embeddings, "text")

    try:

        if not util.check_non_module_in_string(module, question):
            raise CustomError(
                "Sorry I dont have information on these topics."
            )

        # Code that may raise an error
        if not util.check_valid_string(question):
            raise CustomError(
                "An error occured invalid input, please try again later."
            )
        if len(question) <= 0:
            raise CustomError(
                "An error occured empty input, please report try again later."
            )
        # if len(question) >= 1 and len(question) <= 3:
        #     raise CustomError(
        #         "An error occured length less than 3, please report try again later."
        #     )
        if len(question) <= 3 and IS_CHARACTER_JUNK(question):
            raise CustomError(
                "An error occured not a valid input, please report try again later."
            )

        # if you already have an index, you can load it like this
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

        docs = docsearch.similarity_search(query)

        if docs:
            few_shot = docs[0].page_content
        else:
            # Handle the case where docs is empty
            few_shot = "No documents found"
        # print(few_shot)

        def get_schema(_):
            return sqldb.get_table_info()

        def run_query(query):
            return sqldb.run(query)

        # few shot template 1
        prompt = ChatPromptTemplate.from_template(
            SQL_PREFIX + SQL_QUERY_DESCPTION + ASSETS_EXT + FEW_SHOT_EXAMPLE_TEMPLATE)

        verbose = os.environ.get("VComGPTDEBUG", False)
        # print(verbose)
        model = ChatOpenAI(model=AIModel, temperature=temperature, verbose=verbose)
        # print(model)
        sql_response = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | model.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
        )

        # prompt template FINAL_FEWSHOT_ASSET_PROMPT_TEMPLATE
        # ASSETS_EXT +
        prompt_response = ChatPromptTemplate.from_template(
            ASSETS_EXT_001 + FINAL_FEWSHOT_ASSET_PROMPT_TEMPLATE_001)

        full_chain = (
            RunnablePassthrough.assign(
                # schema=get_schema,
                response=lambda x: sqldb.run(x["query"]),
            )
            | prompt_response
            | model
            | StrOutputParser()
        )

        # grab user info with tool.
        callbacks = [LoggingCallbackHander(user_id=user_id)]
        query_users = QueryUserNames(db=sqldb, user_view="AI_UserIDXNameView", callbacks=callbacks)
        user_info = query_users._run(str(user_id))

        # print(user_info)

        topic = module
        dialect = sqldb.dialect
        k = 10
        year = date.today().year,
        baseURL = os.environ.get("BASE_URL")

        sql_query = sql_response.invoke(
            {"CustomerID": customer_id, "question": question, "few_shot_example": few_shot, "k": k, "user_info": user_info,
                "topic": topic, "dialect": dialect, "year": year, "baseURL": baseURL})

        # Remove unwanted data from the query
        sql_query = sql_query.strip().replace("SQL Query: ", "")
        # print("final query --> " + sql_query)
        if str(sql_query).startswith("SELECT") or str(sql_query).startswith("SQL Query: ") or str(sql_query).startswith("DECLARE") or str(sql_query).startswith("declare"):
            ai_response = full_chain.invoke({"question": question, "query": sql_query,
                                            "CustomerID": customer_id, "k": k, "user_info": user_info, "topic": topic, "dialect": dialect, "year": year, "baseURL": baseURL})
        else:
            ai_response = "I am unable to answer your question at this time"
        ai_response = util.get_tabular_format(ai_response)
        await memory.dump_user_memory(user_id, mem)

    except CustomError as ve:
        res_message, error_code = ("Error", 400)  # Unpack error message and code

        if not util.check_non_module_in_string(module, question):
            _unknown = f"Hello! It looks like you've asked something I am not trained on '{question}'.\nPlease ask something related to <b>{module}.</b>"
        if not util.check_valid_string(question):
            _unknown = f"Hello! It looks like you've entered incomplete prompt '{question}.'"
            "Replace keyword between {\"keyword\"}. I'll do my best to assist you."
        if len(question) <= 0:
            _unknown = "Hello! It looks like you've entered nothing. If you have a specific question or need help with"
            "something, please feel free to elaborate, and I'll do my best to assist you."

        if len(question) <= 3 and IS_CHARACTER_JUNK(question):
            _unknown = f"Hello! It looks like you've entered just the letter '{question}.'"
            "If you have a specific question or need help with something, please feel free to elaborate,"
            "and I'll do my best to assist you."

        if IS_LINE_JUNK(question):
            _unknown = "Hello! It looks like you've not entered anything."
            "Can you provide more information or clarify your question? I'd be happy to help with whatever you need."

        # Custom error handling
        print("Error:", ve)
        error_message = {
            aierror.InvalidRequestError: ve,
        }
        ai_response = error_message.get(type(ve), _unknown)
    except Exception as err:
        res_message, error_code = ("Error", 400)  # Unpack error message and code
        if (environment.lower() == "debug"):
            _unknown = "An unknown error occured, please report this or try again later(.) " + \
                "Error reason" + str(err)
        else:
            _unknown = "An unknown error occured, please report this or try again later"

        error_message = {
            aierror.InvalidRequestError: "vChat likely used too many words while processing your request."
            "Try limiting how many results by adding something similar to 'only give me the top 3 results'.",
            aierror.RateLimitError: "vChat has exceeded current quota, please contact admin.",
            aierror.Timeout: "vChat took too long to process your request, try again in a little while.",
            aierror.TryAgain: "Something went wrong processing your request, please try again later.",
            aierror.ServiceUnavailableError: "Something went wrong with" + AIModel + ", "
            "please try again later and report this error if possible.",
            OutputParserException: "vChat ran into an issue parsing some text, try modifying your question.",
            aierror.APIConnectionError: "Something went wrong with " + AIModel + ", "
            "please try again later and report this error if possible.",
            aierror.APIError: "Something went wrong with " + AIModel + ", "
            "please try again later and report this error if possible.",
            aierror.OpenAIError: "Something went wrong with " + AIModel + ", "
            "please try again later and report this error if possible.",
        }
        ai_response = error_message.get(type(err), _unknown)
        print(err)

    # Final response
    response_data = {
        "message": res_message,
        "response": {
            "gptErrorCode": error_code,
            "request": question,
            "module": norm_module,
            "userID": user_id,
            "pyResponse": ai_response,
            "PromptName": promptName,
            "TableorViews": views,
            "ColumnLookup": None,
            "isInternal": internal,
            "CustomerID": customer_id,
            "graphImageURL": graph_image_url,
            "FinalQuery": sql_query,
            "ftModel": AIModel,
            "AIResponse": ai_response,
        }
    }

    return jsonify(response_data), 200, response_headers
