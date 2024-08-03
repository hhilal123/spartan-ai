import os
import openai
kfhdkhhdkhkdkdkdkdkkdkdkdkdkdkd
from dotenv import load_dotenv
load_dotenv('.env')
# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
print("OpenAI API Key:", openai.api_key)
# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Define the path to your data file
data_file_path = r'D:\Nagasai.Works\vGPT\ft_data_model\asset_data_v1.jsonl'
changes = "assets_v46"
file_name = os.path.basename(data_file_path)
print(file_name)
# Read the data from the file
with open(data_file_path, 'rb') as file: pdata = file.read()
# Create a file object with the data
file_obj = openai.File.create(file=data, purpose='fine-tune', user_provided_filename=file_name)
# Alternatively, you can use a URL to the data file
# response = requests.get('https://somesite/mydata.jsonl')
# file_obj = openai.File.create(file=response.content, purpose='fine-tune')
# Print the file
print(file_obj)
# Print the file id
print(file_obj.id)
# gpt-3.5-turbo-0613 (using till v28)
# gpt-3.5-turbo-1106 (recommended from v29)
# gpt-4-0613 (experimental version - from v39)
model = "gpt-3.5-turbo-1106"
create_res = openai.FineTuningJob.create(training_file=file_obj.id, model=model, suffix=changes)
# print create job response
print(create_res)

# print create job response
print("job id -->" + create_res.id + " create job status -->" + create_res.status)
# Retrieve the state of a fine-tune
retreive_res = openai.FineTuningJob.retrieve(create_res.id)
# print retreive job response
print("job id -->" + retreive_res.id + " retreive status -->" + retreive_res.status)