import flask
from flask import Flask, render_template,request,redirect
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField , FloatField , SelectField, SubmitField
from wtforms.validators import  DataRequired, InputRequired , NumberRange
from flask_wtf import Form
import joblib
import pickle
import os
import pandas as pd 
import sklearn


app = Flask(__name__)
Bootstrap(app)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

class Submit(FlaskForm):
    submit = SubmitField("Predict")

class MyForm(FlaskForm):
    drogue = SelectField("drogue : ", choices=['Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','VSA'], validators=[InputRequired()])
    age = SelectField("Age : ", choices=[(-0.95197,"18-24"),(-0.07854,"25-34"),(0.49788,"35-44"),(1.09449,"45-54"),(1.82213,"55-64"),(2.59171,"65+")], coerce= float , validators=[InputRequired()])
    sexe= SelectField("Sexe : ", choices=[(-0.48246,"Homme"), (0.48246,"Femme")], coerce= float , validators=[InputRequired()])
    education= SelectField("Education : ", choices=[(-2.43591,"A quitté l'école avant 16 ans"),(-1.73790,"A quitté l'école à 16 ans"),(-1.43719 ,"A quitté l'école à 17 ans"),(-1.22751,"A quitté l'école à 18 ans"),(-0.61113,"Université"),(-0.05921,"Certificat professionnel ou diplome"),(0.45468,"Diplome d'université"), (1.16365,"Master"),(1.98437,"Doctorat")], coerce= float , validators=[InputRequired()])
    pays= SelectField("Pays : ", choices=[(-0.09765 ,"Australia"),(0.24923,"Canada"),(-0.46841,"New Zealand"),(-0.28519,"Other"),(0.21128,"Republic of Ireland"),(0.96082,"UK"),(-0.57009,"USA")], coerce= float , validators=[InputRequired()])
    ethnicity= SelectField("Ethnicity : ", choices=[(-0.50212,"Asian"),(-1.10702,"Black"),(1.90725,"Mixed-Black/Asian"),(0.12600,"Mixed-White/Asian"),(-0.22166,"Mixed-White/Black"),(0.11440,"Other"),(-0.31685,"White")], coerce= float , validators=[InputRequired()])
    Nscore = FloatField("Nscore", validators=[DataRequired(),NumberRange(min=-3.46436, max=3.27393)])
    Escore = FloatField("Escore", validators=[DataRequired(),NumberRange(min=-3.27393, max=3.27393)])
    Oscore = FloatField("0score", validators=[DataRequired(),NumberRange(min=-3.27393, max= 2.90161)])
    Ascore = FloatField("Ascore", validators=[DataRequired(),NumberRange(min=-3.46436, max=3.46436)])
    Cscore = FloatField("Cscore", validators=[DataRequired(),NumberRange(min=-3.46436, max=3.46436)])
    Impulsive = FloatField("Impulsive", validators=[DataRequired(),NumberRange(min=-2.55524, max=2.90161)])
    SS = FloatField("SS", validators=[DataRequired(),NumberRange(min=-2.07848, max=1.92173 )])
    Submit = SubmitField("Submit")



@app.route("/",methods=["GET","POST"])

def Informations_Form():
    myform = MyForm()
    if request.method=="POST" and myform.validate_on_submit:
        model = pickle.load(open('models/final_prediction_'+myform.drogue.data+'.pickle', 'rb'))
        result = model.predict(pd.DataFrame([[myform.age.data,myform.sexe.data,myform.education.data,myform.pays.data
        ,myform.ethnicity.data,myform.Nscore.data,myform.Escore.data,myform.Oscore.data,myform.Ascore.data,myform.Cscore.data
        ,myform.Impulsive.data,myform.SS.data]],columns=  ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']))
        return render_template('resultats.html',result = result)
    return render_template('index.html',form=myform)
    



if __name__== "__main__":
    app.run(port=5000,debug=True)