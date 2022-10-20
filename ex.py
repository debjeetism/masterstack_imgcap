
# from flask import Flask,render_template,request
# import captionit
# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return render_template("main.html")

# @app.route("/main", methods=['GET','POST'])
# def main():
#      if request.method=='POST':
        
#         image= request.files['userfile']
#         path= "./static/{}".format(image.filename)
#         image.save(path)
#         caption=captionit.caption_this_image(path)
#      return render_template("main.html",caption=caption)
# app.run(debug=True)  
# 
# 
# from email.mime import image
from importlib.resources import path
from operator import methodcaller, truediv
from pickle import TRUE
import captionit
import imp
import re
from sqlite3 import Cursor
from flask import Flask, render_template,  request, redirect, url_for,session
from flask_mysqldb import MySQL
import mysql.connector
import MySQLdb.cursors
import re
app = Flask(__name__)
app.secret_key='secret_key'
app.config['MYSQL_HOST']= 'localhost'
app.config['MYSQL_USER']= 'root'
app.config['MYSQL_PASSWORD']= '123456789'  
app.config['MYSQL_DB']= 'login'
mysql =MySQL(app)



@app.route("/")
def home(): 
    url="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.news-medical.net%2Fhealth%2FHow-Does-a-Cornea-Transplant-Work.aspx&psig=AOvVaw1re-X4WEvAgvXV-GEMB8iS&ust=1666340293265000&source=images&cd=vfe&ved=0CA0QjRxqFwoTCPDThbyv7voCFQAAAAAdAAAAABAW"

    return render_template('index.html',image_url=url)

@app.route("/login")
def about():
    return render_template('login.html')

@app.route("/main")
def main():
    return render_template("main.html") 

@app.route("/upload", methods=['GET','POST'])
def upload():
    if request.method=='POST':

        image= request.files['userfile']
        path= "./static/{}".format(image.filename)
        image.save(path)
        msg='hello'
        caption = captionit.caption_this_image(path)
        # combo={
        #     'image': path,
        #     'caption': caption
        # }
    return render_template("main.html",caption=caption)
    # return render_template("main.html")
    
  
   


@app.route("/submit", methods=['GET','POST'])
def authentication():
    msg=''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email=request.form['email']
        password=request.form['password']
        cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * from Users where email='" + email + "' and password='" + password + "'")
       
        Users = cursor.fetchone() 
        if Users:
            session['loggedin']= True
            session['id']=Users['id']
            session['username']= Users['username']
            msg="you have logged in successfully"
            dict={
                'user':  session['username'],
                'note': msg
            }
            return render_template('main.html',obj=dict)
            # return redirect(url_for('main'))
            
        else:
            msg='Incorrect username/password' 

    return render_template('login.html',msg=msg)   
@app.route("/logout")
def logout():
    session.clear() 
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template('signup.html')

    
@app.route("/signup", methods=['GET' ,'POST'])
def register():
    msg=''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
         username = request.form.get('username')
         password = request.form.get('password')
         email = request.form.get('email')
         cursor= mysql.connection.cursor(MySQLdb.cursors.DictCursor)
         cursor.execute('SELECT * FROM Users WHERE username = %s', (username,))
         data = cursor.fetchone()
         if data:
            msg='account already exists!'
         else:
            cursor.execute('INSERT INTO Users VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!' 
        
    
    elif request.method=='POST':
        msg= "plz fill out the form"
    return render_template('signup.html',msg=msg)
   

app.run(debug=TRUE)  