from datetime import timedelta

from flask import Flask, flash, redirect, render_template, request, session, url_for

app = Flask(__name__)

@app.route("/chat", methods=["POST", "GET"])
def chat():
    recent_chat = " "
    if request.method == "POST":
        chat = request.form["chat"]
        recent_chat = chat
    return render_template("chat.html", chat=recent_chat)


if __name__ == "__main__":
    app.run(debug=True)
