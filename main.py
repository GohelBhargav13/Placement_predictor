from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import joblib
import os



app = Flask(__name__)

# set-up the CORS
CORS(app, 
     resources={
         r"/*": {
             "origins":["http:localhost:3000"]
         }
     },
    allow_headers=["Content-Type","Authorization"],
    methods=["GET","POST","OPTIONS","DELETE"],
)

# importing the custom model using the joblib
model = joblib.load("model.pkl")

@app.route("/")
def home(): 
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict(): 
    form = request.form
    errors = {}

    name = form.get("name", "").strip()
    cgpa = form.get("cgpa", "").strip()
    backlogs = form.get("backlogs", "").strip()
    skills = form.get("skills", "").strip()
    internships = form.get("internships", "").strip()
    projects = form.get("projects", "").strip()

    if not cgpa:
        errors["cgpa"] = "CGPA is required."
    if not name:
        errors["name"] = "NAME is required."
    if not backlogs:
        errors["backlogs"] = "BACKLOGS is required."
    if not skills:
        errors["skills"] = "SKILLS is required."
    if not internships:
        errors["internships"] = "INTERNSHIP is required."
    if not projects:
        errors["projects"] = "PROJECTS is required."

    if errors:
        return render_template("index.html", errors=errors, form=form)

    # now used the model for the prediction
    student = pd.DataFrame([{
            "cgpa": cgpa, 
            "backlogs": backlogs, 
            "projects": projects, 
            "internships": internships, 
            "skills": skills
    }])

    prob = model.predict_proba(student)[0][1]
    result = model.predict(student)[0]


    return render_template("result.html", summary=form, placed=result, percent=round(prob*100, 2), user_name=name)

@app.route("/result")
def result():
    return render_template("result.html", placed=1, percent=80)

if __name__ == "__main__":  
    app.run(port=int(os.getenv("PORT", 5000)))
