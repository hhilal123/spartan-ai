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

# instantiating important function calls
app = FastAPI()
util = UtilFunctions()

# system wide variable
environment = os.environ.get("ENVIRONMENT", "")
memory_len = int(os.environ.get("MEMORY_LENGTH", "3"))  # Default to 5
temperature = float(os.environ.get("TEMPERATURE", "0.2"))

load_dotenv(".env") 

# api connection variables
OPENAI_API_VERSION = '2023-12-01-preview'
model = os.getenv("AZURE_OPENAI_MODEL")
verbose = False
deployment_id = os.getenv("AZURE_EMBEDDING_DEPLOYID")
deployment_name = os.getenv("AZURE_DEPLOYID")
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_version = OPENAI_API_VERSION

# setting up Azure Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=deployment_id,
    openai_api_key=openai.api_key,
    api_version=openai.api_version,
)

# setting up Azure LLM
llm = AzureChatOpenAI(azure_deployment=deployment_name,
                      model=model,
                      temperature=temperature,
                      streaming=True,
                      max_tokens=4000,
                      callback_manager=BaseCallbackManager([
                          StreamingStdOutCallbackHandler()]),
                      openai_api_key=openai.api_key,
                      verbose=verbose,
                      openai_api_version=OPENAI_API_VERSION,
                      top_p=1,
                      frequency_penalty=0.0,
                      presence_penalty=0.0)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    context_key="context",
    return_messages=True,
    output_key="output",
)

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False,
)


for item in (model):
    if not item:
        raise ValueError("Model not defined")



class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False
    finalResponse: str = ""

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""


async def run_call(query: str, stream_it: AsyncCallbackHandler):
    """
    Run a call with the given query and stream the results using the provided callback handler.

    Parameters:
        query (str): The query string to be passed to the agent for processing.
        stream_it (AsyncCallbackHandler): The callback handler to stream the results.

    Returns:
        None
    """
    # assign callback handler
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    await agent.acall(inputs={"input": query})


class Query(BaseModel):
    Request: str



async def generate_json_data(module: Optional[str] = None):
    data = {"module": module}
    return json.dumps(data).encode("utf-8")


async def create_gen(query: str, stream_it: AsyncCallbackHandler, callback=None):
    """
    Asynchronously creates a generator for processing a given query and streaming the results using the provided callback handler.

    Args:
        query (str): The query string to be processed.
        stream_it (AsyncCallbackHandler): The callback handler to stream the results.
        callback (Optional[Callable]): An optional callback function to be executed on the response.

    Yields:
        Any: The next token in the streamed response.

    Returns:
        None
    """
    task = asyncio.create_task(run_call(query, stream_it))
    async for token in stream_it.aiter():
        if not token:
            break
        yield token
    await task

    if callback:

        async def handle_response_async(token):  # Make callback function async
            try:
                result = await callback(
                    token
                )  # Await potential async callback function
                return result
            except Exception as e:
                print(
                    f"Error during callback execution: {e}"
                )  # Handle exception in callback

        # Run callback function in a separate task for proper async handling
        await asyncio.create_task(handle_response_async(token))


async def handle_response(response):  # Example callback function (can be sync or async)
    print(f"Final response: {response}")


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
    re.compile(r"https?://localhost(:\d+)?"),  # Regex for localhost with optional port
]

# Enable CORS
app.add_middleware(
    RegexCORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    UserId: int
    Request: str
    Environment: str
    GUID: str


class ResponseMessage(BaseModel):
    Request: str
    UserId: int
    CustomerID: int


@app.post("/api/dls-chat/v1")
async def chat(message: Message, request: Request, background_tasks: BackgroundTasks):

    try:
        user_id = int(message.UserId)
        customer_id = int(message.CustomerID)
    except TypeError:
        return {
            "error": "Invalid request, could not find valid integer 'UserId' or 'CustomerID'"
        }

    question = message.Request.strip()
    referer = request.headers.get("Origin")

    if referer is None:
        internal = "false"
    elif referer.find("localhost") == -1:
        internal = "false"
    else:
        internal = "true"


    standardised_question = ""
    # print(firstname, lastname)
    query = "Question: " + question
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

        # start : standardised_question based on memory
        chat_history = memory.load_memory_variables({})["chat_history"]
        contextualize_q_system_prompt = """
            Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history or If the questions is about individual. \
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is. \
        
            Chat history: {chat_history}
        """

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        if len(chat_history):
            standardised_question = await contextualize_q_chain.ainvoke(
                {
                    "chat_history": chat_history,
                    "question": question,
                }
            )

        if standardised_question == "" or standardised_question == " ":
            standardised_question = question
        else:
            chat_history.extend(
                [
                    HumanMessage(content=question),
                    AIMessage(content=standardised_question),
                ]
            )


        error_code = 200
        res_message = "Success"
        year = date.today().year
        baseURL = os.environ.get("BaseURL")
        
        prompt_response = ChatPromptTemplate.from_template(
            #this is where we put prompting
        )

        prompt_question = prompt_response.format(
            question=standardised_question,
            baseURL = baseURL,
        )

        stream_it = AsyncCallbackHandler()
        gen = create_gen(prompt_question, stream_it, callback=handle_response)

        return StreamingResponse(gen, media_type="text/event-stream")

    # Consolidated multiple exception blocks into a single except block for improved readability and maintainability.
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

        return StreamingResponse(
            io.StringIO(ai_response),
            status_code=error_code,
            media_type="text/event-stream",
        )