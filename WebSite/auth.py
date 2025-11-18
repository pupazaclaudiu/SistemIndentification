from flask import Blueprint

auth1 = Blueprint('auth', __name__)


@auth1.route('/login')
def login():
    return "<h1>Login Page</h1>"