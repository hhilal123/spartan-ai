import os

import openai
from dotenv import load_dotenv
# Colin is a big boy
# OpenAI API key
openai.api_key = os.environ.get(OPENAI_API_KEY)
openai.Model = os.environ.get(OPENAI_MODEL)
