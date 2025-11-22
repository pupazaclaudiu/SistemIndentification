from flask import Blueprint, render_template

auth1 = Blueprint('auth', __name__)


@auth1.route('/login')
def login():
    return render_template("login.html",text="Testing",user="Tim",boolean=False)

@auth1.route('/logout')
def logout():
    return "<h1>Logout Page</h1>"

@auth1.route('/signUp')
def sign_up():
    return render_template("signUp.html")