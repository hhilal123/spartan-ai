from datetime import timedelta

from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "placeholder123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.permanent_session_lifetime = timedelta(hours=1)

db = SQLAlchemy(app)


class users(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column("name", db.String(100))
    student_id = db.Column("student_id", db.Integer)

    def __init__(self, name, student_id) -> None:
        self.name = name
        self.student_id = student_id


@app.route("/")
def home():
    return render_template("index.html", content="Hudson")


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        session.permanent = True
        user = request.form["nm"]
        student_id = request.form["s_id"]
        session["user"] = user
        session["student_id"] = student_id

        found_user = users.query.filter_by(name=user, student_id=student_id).first()

        if found_user:
            flash(f"Welcome Back, {user}")
        else:
            usr = users(user, student_id)
            db.session.add(usr)
            db.session.commit()

        return redirect(url_for("chat"))
    if "user" in session:
        return redirect(url_for("chat"))
    return render_template("login.html")


@app.route("/chat", methods=["POST", "GET"])
def chat():
    recent_chat = " "
    if request.method == "POST":
        chat = request.form["chat"]
        recent_chat = chat
    if "user" in session:
        user = session["user"]
        return render_template("chat.html", user=user, chat=recent_chat)
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    """test method"""
    if "user" in session:
        user = session["user"]
        flash(f"You have been logged out,{user}", "info")
    session.pop("user", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
    with app.app_context():
        db.create_all()
