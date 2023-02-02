print('hello')
import streamlit as smt
import pickle as pkl
import numpy as np
import pandas as pd
pickle_in1 = open('rf.pkl', 'rb')
rf = pkl.load(pickle_in1)
smt.title("CARDIOVASCULAR RISK PREDICTION")


def prediction1(age,sex,is_smoking,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose):
    prediction1 = rf.predict(
        [[age,sex,is_smoking,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]])
    print(prediction1)
    return prediction1

def main():
    smt.title("CARDIOVASCULAR RISK PREDICTION")
    html_temp = """  
       <div style = "background-colour: #FFFF00; padding: 16px">  
       <h1 style = "color: #000000; text-align: centre; "> Cardiovascular Risk Prediction Classifier ML App   
        </h1>  
       </div>  
       """
    smt.markdown(html_temp, unsafe_allow_html=True)
    age = smt.text_input("age ", " Type Here")
    sex= smt.text_input("sex", " Type Here")
    is_smoking = smt.text_input("is_smoking", " Type Here")
    cigsPerDay = smt.text_input("cigsPerDay", " Type Here")
    BPMeds = smt.text_input("BPMeds", " Type Here")
    prevalentStroke = smt.text_input("prevalentStroke", " Type Here")
    prevalentHyp = smt.text_input("prevalentHyp", " Type Here")
    diabetes = smt.text_input("diabetes", " Type Here")
    totChol = smt.text_input("totChol", " Type Here")
    sysBP = smt.text_input("sysBP", " Type Here")
    diaBP = smt.text_input("diaBP", " Type Here")
    BMI= smt.text_input("BMI", " Type Here")
    heartRate = smt.text_input("heartRate", " Type Here")
    glucose = smt.text_input("glucose", " Type Here")
    result = " "
    if smt.button("Predict"):
        result = prediction1(age,sex,is_smoking,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose)
    smt.success('The output of the above is {}'.format(result))


if __name__ == '__main__':
    main()


