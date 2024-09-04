import os
import asyncio
import io
import json
import re
import time
import agent
from datetime import date
from difflib import IS_LINE_JUNK
from functools import lru_cache
from typing import Any, Optional
import openai as aierror
import openai
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import LLMResult, OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel
from agent.util import CustomError, UtilFunctions
from agent.chatprompt import (
    RESPONSE_PROMPT
)


# loads local environment
load_dotenv(".env") 

# instantiating important function calls
app = FastAPI()
util = UtilFunctions()

# system wide variables
environment = os.environ.get("ENVIRONMENT", "")
memory_len = int(os.environ.get("MEMORY_LENGTH", "3"))  # Default to 5
temperature = float(os.environ.get("TEMPERATURE", "0.2"))


# following defines llm variables
verbose = False
model = os.getenv("AZURE_OPENAI_MODEL")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")



# setting up llm through azure
llm = AzureChatOpenAI(azure_deployment=deployment_name,
                      model=model,
                      temperature=temperature,
                      streaming=True,
                      max_tokens=100,
                      callback_manager=BaseCallbackManager([
                          StreamingStdOutCallbackHandler()]),
                      openai_api_key=openai.api_key,
                      verbose=verbose,
                      openai_api_version=openai.api_version,
                      top_p=1,
                      frequency_penalty=0.0,
                      presence_penalty=0.0)


# defines a memory based on the context of the chat
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    context_key="context",
    return_messages=True,
    output_key="output",
)

# defines the llm agent
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False,
    handle_parsing_errors=True,
)

# Handles the output from asynchronous token generation
class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""  # Initializes content to accumulate tokens
    final_answer: bool = False  # Flag to indicate if the final answer has been reached

    def __init__(self) -> None:
        super().__init__()

    # Handles each new token generated by the LLM
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token

        # If final answer has been reached, process incoming tokens
        if self.final_answer:
            # Filter out certain tokens for output handling
            if token not in ['"', "}"]:
                self.queue.put_nowait(token)  # Add the token to the output queue for asynchronous handling
        # Detect when the final answer starts
        elif "Final Answer" in self.content:
            self.final_answer = True  # Set the flag to indicate final answer is in progress
            self.content = ""  # Clear the content for the final answer processing

    # Handles the end of LLM's response
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""  # Clear the content after processing the final answer
            self.final_answer = False  # Reset the final answer flag
            self.done.set()  # Signal that the processing is complete
        else:
            self.content = ""  # Reset content if final answer wasn't reached

# Function to run the query and stream the response
async def run_call(query: str, stream_it: AsyncCallbackHandler):
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    await agent.acall(inputs={"input": query})  # Asynchronously call the agent with the query input

# Base model for the query request
class Query(BaseModel):
    Request: str

# Function to create and handle the generation of responses from the LLM
async def create_gen(query: str, stream_it: AsyncCallbackHandler, callback=None):
    task = asyncio.create_task(run_call(query, stream_it))  # Create an async task for the query execution
    response: any = ""

    # Asynchronously iterate over the generated tokens
    async for token in stream_it.aiter():
        if not token:
            break  # Stop if no more tokens are generated
        yield token
        response += token  # Accumulate the tokens into the response

    await task

    #  async function to handle callback
    if callback:
        async def handle_response_async(response):
            try:
                # Await the callback and process the response
                result = await callback(response)
                return result
            except Exception as e:
                print(f"Error during callback execution: {e}")

        # Execute the callback asynchronously as a separate task
        await asyncio.create_task(handle_response_async(response))

# Example callback function to handle the response
async def handle_response(response):  # Example of a callback function (can be sync or async)
    print("")


# handles the CORS middleware based on a regular expression defined
class RegexCORSMiddleware(CORSMiddleware):
    def __init__(
        self,
        app: FastAPI,
        allow_origins: list = ["*"],
        allow_origin_regex: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        expose_headers: list = None,
        allow_credentials: bool = False,
        max_age: int = 600,
    ):
        super().__init__(
            app,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            expose_headers=expose_headers,
            allow_credentials=allow_credentials,
            max_age=max_age,
        )
        self.allow_origin_regex = allow_origin_regex or []

    async def is_allowed_origin(self, origin: str) -> bool:
        if "*" in self.allow_origins:
            return True
        for regex in self.allow_origin_regex:
            if await regex.match(origin):
                return True
        return False

# Define allowed origins using regular expressions
allowed_origins = [
    re.compile(r"https?://localhost(:\d+)?"),
    re.compile(r"https?://127.0.0.1(:\d+)?"),  # Regex for localhost with optional port
]

# Enable CORS
app.add_middleware(
    RegexCORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# formatting for input messages 
class Message(BaseModel):
    UserId: int
    Request: str
    Environment: str
    FirstName: str
    LastName: str
    UserId: int
    GUID: str


# main chat function for the managament and processing of the user input
@app.post("/api/dls-chat/v1")
async def chat(message: Message, request: Request, background_tasks: BackgroundTasks):

    try:
        user_id = int(message.UserId)
    except Exception as err:
        # Log the error type and message for debugging
        print(f"Error: {type(err)}, {str(err)}")
        print(f"An unexpected error occurred: {type(err).__name__}, {str(err)}")
    
    
    question = message.Request.strip()
    print(f"Question:{question}")

    standard_question = ""
    query = "Question: " + question
    # Error handling
    try:
        start_time = time.time()
        if not util.check_relevance(question):
            error = (
                "Hello! It looks like you've asked something I am not trained on  "
                f"'{question}'\n"
                f"Please ask something related to editing or providing critique.</b> "
                f"I'll do my best to assist you."
            )
            raise CustomError(error, error_type="CustomError")
        # Code that may raise an error
        if not util.check_valid_string(question):
            error = (
                "Hello! It looks like you've entered incomplete prompt "
                f"'{question}'\n"
                f'Replace the keyword "{{keyword}}" with an relevant term. '
                f"I'll do my best to assist you."
            )
            raise CustomError(error, error_type="CustomError")
        if len(question) <= 1:
            error = (
                "Hello! It looks like you've entered nothing. "
                f"If you have a specific question or need help with  "
                f"something, \n Please feel free to elaborate, and I'll do my best to assist you. "
            )
            raise CustomError(error, error_type="CustomError")
        if question.lower() == "hi" or question.lower() == "hello":
            error = "Hello! How can I assist you today? "
            raise CustomError(error, error_type="CustomError")

        if len(question) <= 3 or util.is_junk_string(question):
            error = (
                "Hello! It looks like you've entered just the letters "
                f"'{question}'\n"
                f"If you have a specific question or need help with something, "
                f"Please feel free to elaborate, and I'll do my best to assist you."
            )
            raise CustomError(error, error_type="CustomError")

        if IS_LINE_JUNK(question):
            error = (
                "Hello! It looks like you've not entered anything. "
                f"Can you provide more information or clarify your "
                f"question? \n I'd be happy to help with whatever you need."
            )
            raise CustomError(error, error_type="CustomError")

        # standardizes the question based on memory
        chat_history = memory.load_memory_variables({})["chat_history"]
        contextualize_q_system_prompt = """
            Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history or If the questions is about individual. \
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is. \
        
            Chat history: {chat_history}
        """

        # contextualize based on prompting
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        # looks back in chat history
        if len(chat_history):
            standard_question = await contextualize_q_chain.ainvoke(
                {
                    "chat_history": chat_history,
                    "question": question,
                }
            )

        # defines the star
        if standard_question == "" or standard_question == " ":
            standard_question = question
        else:
            chat_history.extend(
                [
                    HumanMessage(content=question),
                    AIMessage(content=standard_question),
                ]
            )

        error_code = 200
        res_message = "Success"
        year = date.today().year
        baseURL = os.environ.get("BaseURL")
        
        prompt_response = ChatPromptTemplate.from_template(
            RESPONSE_PROMPT
        )

        prompt_question = prompt_response.format(
            question=standard_question,
            baseURL = baseURL,
        )

        stream_it = AsyncCallbackHandler()
        gen = create_gen(prompt_question, stream_it, callback=handle_response)
        return StreamingResponse(gen, media_type="text/event-stream")
        
    # Consolidated multiple exception blocks into a single except block for improved readability.
    except Exception as err:

        # Custom error handling
        print("Error:", type(err))
        error_message = {
            aierror.ConflictError: (
                "Issue connecting to our services, please try again later and report this error if possible."
            ),
            aierror.NotFoundError: (
                "Requested resource does not exist, please try again later and report this error if possible."
            ),
            aierror.APIStatusError: (
                "Something went wrong processing your request, please try again later."
            ),
            aierror.AuthenticationError: (
                "Your API key or token was invalid, expired, or revoked. Please try again later and report this error if possible."
            ),
            aierror.InternalServerError: (
                "Something went wrong processing your request, please try again later."
            ),
            aierror.PermissionDeniedError: (
                "No access to the requested resource, please try again later."
            ),
            aierror.UnprocessableEntityError: (
                "Something went wrong processing your request, please try again later."
            ),
            aierror.BadRequestError: (
                "vChat likely used too many words while processing your request. Try limiting how many results by adding something similar to 'only give me the top 3 results'."
            ),
            aierror.RateLimitError: (
                "vChat has exceeded the current quota, please try again after some time."
            ),
            aierror.APITimeoutError: (
                "vChat took too long to process your request, try again in a little while."
            ),
            OutputParserException: (
                "vChat ran into an issue parsing some text, try modifying your question."
            ),
            aierror.APIConnectionError: (
                f"Something went wrong with the OpenAI API, please try again later and report this error if possible."
            ),
            aierror.APIError: (
                f"Something went wrong with OpenAI API, please try again later and report this error if possible."
            ),
            aierror.OpenAIError: (
                f"Something went wrong with OpenAI API, please try again later and report this error if possible."
            ),
        }
        res_message, error_code = (str(err), 400)  # Unpack error message and code
        ai_response = error_message.get(type(err), res_message)

        if ai_response is None:
            if environment.lower() == "local":
                ai_response = (
                    "An unknown error occured, please report this or try again later(.) "
                    + "Error reason "
                    + str(err)
                )
            else:
                ai_response = "An unknown error occured while connecting to database, please report this and try again!!"

        if environment.lower() == "local":
            ai_response = (
                ai_response + "<br> <b> Transcation ID - " + message.GUID + " </b>"
            )

        return StreamingResponse(io.StringIO(ai_response), status_code=error_code, media_type="text/event-stream")
    
