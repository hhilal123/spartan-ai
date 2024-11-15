from flask import Flask, render_template

app = Flask(__name__)


@app.route("/chat")
def index():
    return render_template("chat.html")


@app.post("/chat")
def chat():
    #need to implement logic here

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
