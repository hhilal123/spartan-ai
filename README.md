DLS-Chat

This is a project started by De La Salle High School students in Concord, California. The goal is to build an effect chatbot that leverages OpenAI's finetuning API to be able to create a healthy student interactive AI chat.

# Setup

Follow these instructions to setup the project on your machine:

**1. Clone Repo**

```zsh
git clone https://github.com/hhilal123/spartan-ai
```

**2. Install Requirements**

Navigate to the project folder and set up Python virtual environment.

```zsh
python3 -m venv .venv
```

Activate virtual environment.

```zsh
source .venv/bin/activate
```

Install requirements

```zsh
pip install -r requirements.txt
```

**3. Add Requirements to the .env**

Get relevant keys and info from project manager.

**4. Run locally**

Run with flask in the terminal.

```zsh
flask --app app run
```
