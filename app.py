from flask import Flask, request,render_template
import numpy as np
import pickle
from json import JSONEncoder

reg = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__,template_folder='templates')


@app.route("/")
def index():
    return render_template("index.html")

# Home page route
@app.route("/home")
def home():
    return render_template("home.html")

# Web clasification route
@app.route("/clasify",methods=["POST"])
def clasify():
    sm = float(request.form['sm'])
    person = float(request.form['person'])
    issue = float(request.form['issues'])
    sc = float(request.form['sc'])
    life = float(request.form['life'])
    depression = float(request.form['depression'])
    arr = np.array([[sm,person,issue,sc,life,depression]])
    predection = reg.predict(arr)
    return render_template("clasify.html",data = predection)

# Suggations route
@app.route("/suggations")
def suggations():
    return render_template("suggations.html")

# Route for API
@app.route("/api/clasify",methods=["POST"])
def api_clasify():
    sm = float(request.form['sm'])
    person = float(request.form['person'])
    issue = float(request.form['issues'])
    sc = float(request.form['sc'])
    life = float(request.form['life'])
    depression = float(request.form['depression'])
    arr = np.array([[sm,person,issue,sc,life,depression]])
    predection = reg.predict(arr)
    data = {}
    data['value'] = int(predection)
    return data

if __name__ == "__main__":
    app.run()