from flask import Flask
from dotenv import load_dotenv
import os
from .Views import views
from .auth import auth1

def create_app():
    app=Flask(__name__)
    load_dotenv()
    app.config['SECRET_KEY']=os.getenv('SECRET_KEY')

    app.register_blueprint(views,url_prefix='/')
    app.register_blueprint(auth1,url_prefix='/')

    return app

