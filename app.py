from datetime import timedelta
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from flask import Flask, flash, redirect, render_template, request, session, url_for

app = Flask(__name__)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

@app.route("/chat", methods=["POST", "GET"])
def chat():
    recent_chat = " "
    if request.method == "POST":
        chat = request.form["chat"]
        user_input = tokenizer.encode(chat, return_tensors="pt")
        output = model.generate(user_input, max_length=50)
        recent_chat = tokenizer.decode(output[0], skip_special_tokens=True)
    return render_template("chat.html", chat=recent_chat)


if __name__ == "__main__":
    app.run(debug=True)
