from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, flash, redirect, render_template, request, session, url_for
from chatprompt import RESPONSE_PROMPT
from flask_session import Session

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/", methods=["POST", "GET"])
def home():
    button_message = "Try It!"
    if session:
        button_message = "Resume"
    if request.method == "POST":
        return redirect(url_for("chat"))
    return render_template("home.html", button_message=button_message)

@app.route("/chat", methods=["POST", "GET"])
def chat():
    ai_response = ""
    user_prompt = ""
    chat_history = []
    if request.method == "POST":
        user_prompt = request.form["chat"]
        messages = [
            {"role": "system", "content": RESPONSE_PROMPT},
            {"role": "user", "content": user_prompt}
            ]
        outputs = pipe(
            messages,
            max_new_tokens=512,
        )
        ai_response = outputs[0]["generated_text"][-1]['content']
        chat_history = session.get("chat_history", [])
        chat_history.append({"user": user_prompt, "ai": ai_response})
        session["chat_history"] = chat_history
    else: 
        chat_history = session.get("chat_history", [])


    return render_template("chat.html", chat_history=chat_history)

@app.route("/logout", methods=["GET"])
def logout():
    session.pop("chat_history", default=None)
    return "<h1>Logged out!</h1>"

if __name__ == "__main__":
    app.run(debug=True)
