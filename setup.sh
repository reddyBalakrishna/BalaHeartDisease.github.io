#! /bin/bash

echo "Setting up new flask package layout"

echo -n "Application name: "; read APPNAME

if [ "x$APPNAME" != "x" ]; then

  cd ~
  HOME=`echo $HOME`
  HOME="$HOME/flask_projects/"
  if [ ! -d $HOME ]; then
    mkdir $HOME
  fi

  APPENV="$HOME/$APPNAME/"
  APPHOME="$HOME/$APPNAME/app"

  cd $HOME

  # create virtualenv
  TMP=`which virtualenv`
  RET=$?
  if [ $RET == 0 ]; then 
    if [ ! -d $APPENV ]; then
      if [ ! -d $APPHOME ]; then
        virtualenv --system-site-packages $APPENV
        if [ ! -d $APPHOME ]; then
          mkdir -p $APPHOME 
        fi
      fi
    else
      echo "$APPNAME already exists. Exiting."
      exit 1
    fi
  else
    echo "You need to install virtualenv: pip install virtualenv"
    echo "Exiting."
    exit 1
  fi

# install python modules required within our new environment
  cd $APPENV

# create requirements.txt
  cat > requirements.txt <<EOF
flask==0.9
sqlalchemy==0.7.9
flask-sqlalchemy==0.16
sqlalchemy-migrate==0.7.2
flask-wtf==0.8.4
pytz==2013b
EOF

  source bin/activate
  pip install -r $APPENV/requirements.txt >> /dev/null 2>&1
  deactivate

  if [ $? == 0 ]; then
    echo "Pre-requisites for flask installed."
  fi

  cd $APPHOME

  for i in _init_ views forms models; do 
    touch $i.py
    cat > _init_.py <<EOF
import os
from config import basedir
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
app = Flask(_name_)
app.config.from_object('config')
db = SQLAlchemy(app)
from app import views, models
EOF

  done
  cd $APPHOME
  mkdir static templates
  cd templates
  for i in base.html index.html; do
    touch $i
  done
  cd $APPHOME
  cd static
  for i in base.css favicon.ico; do
    touch $i
  done
  cd $APPENV
  cat > run.py <<EOF
#!/usr/bin/env python
from app import app
app.run(debug= True)
EOF
  chmod +x run.py
  cat > config.py <<EOF
import os
basedir = os.path.abspath(os.path.dirname(_file_))
CSRF_ENABLED = True
SECRET_KETY = 'super-secret-key'
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_migrations')
EOF

echo "* VirtualEnv and Base App Tempalte installed *"
echo "* VirtualEnv: $APPENV *"
echo "* App: $APPHOME *"
echo "* To run server: cd $APPENV; source bin/activate; ./run.py *"

else
  echo Application name required.
  echo Exiting.
  exit 1
fi
