import os
import openai
from dotenv import load_dotenv

load_dotenv('.env')

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
model = os.environ.get("OPENAI_MODEL")
print("OpenAI API Key:", openai.api_key)

# Define the path to your data file
data_file_path = 'training/schema.jsonl'
file_name = os.path.basename(data_file_path)
with open(data_file_path, 'rb') as file: data = file.read()


# Create an OpenAI file object with the data
file_obj = openai.File.create(file=data, purpose='fine-tune', user_provided_filename=file_name)
print(file_obj)