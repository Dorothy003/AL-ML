from flask import Flask,render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route('/')
@app.route('/Home')
def home():
    return render_template('home.html')
                           
@app.route('/Addstudent')
def Addstudent():
    return render_template('Addstudent.html')

@app.route('/Addsubject')
def Addsubject():
    return render_template('Addsubject.html')

@app.route('/Attendancelog')
def Attendancelog():
    return render_template('Attendancelog.html')


if __name__ == "__main__":
    app.run(debug=True)