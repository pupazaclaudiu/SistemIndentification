from flask import Blueprint

auth1 = Blueprint('auth', __name__)


@auth1.route('/login')
def login():
    return "<h1>Login Page</h1>"

@auth1.route('/logout')
def logout():
    return "<h1>Logout Page</h1>"

@auth1.route('/signUp')
def sign_up():
    return "<h1>Sign Up Page</h1>"