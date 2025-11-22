from flask import Flask
from dotenv import load_dotenv
import os
from flask_sqlalchemy import SQLAlchemy

load_dotenv()
db=SQLAlchemy()
DB_NAME=os.getenv("DB_NAME")

def create_app():
    app=Flask(__name__)
    app.config['SECRET_KEY']=os.getenv('SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI']=f"sqlite:///{DB_NAME}"
    db.init_app(app)

    from .views import views
    from .auth import auth1

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth1,url_prefix='/')

    from .models import User, Note

    create_database(app)

    return app

def create_database(app):
    if not os.path.exists("website/" + DB_NAME):
        with app.app_context():
            db.create_all()
        print("Created database!")

