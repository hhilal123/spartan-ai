import os

import openai
from dotenv import load_dotenv

#OpenAI API key
openai.api_key = os.environ.get(OPENAI_API_KEY)
openai.Model = os.environ.get(OPENAI_MODEL)
