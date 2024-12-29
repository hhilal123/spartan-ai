from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, flash, redirect, render_template, request, session, url_for
from chatprompt import RESPONSE_PROMPT

app = Flask(__name__)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/", methods=["POST", "GET"])
def chat():
    display = ""
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
        display = outputs[0]["generated_text"][-1]['content']
    return render_template("chat.html", chat=display)


if __name__ == "__main__":
    app.run(debug=True)
