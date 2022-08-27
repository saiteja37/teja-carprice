from flask import Flask,render_template,request
import  pandas as pd
import numpy as np
import pickle
import sklearn

model=pickle.load(open("teja.pkl","rb"))

app=Flask(__name__)

df=pd.read_csv("Quikr_car.csv")
@app.route("/")
def index():
    df["Name"]=df["Name"].str.split(" ").str.slice(0,3).str.join(" ")
    companies=sorted(df["Company"].unique())
    car_models=sorted(df["Name"].unique())
    year=sorted(df["Year"].unique())
    fuel_types=sorted(df["Fuel_type"].unique())
    print('sklearn: {}'.format(sklearn.__version__))
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_types)

@app.route("/predict",methods=["POST"])
def predict():
    company=request.form.get("company")
    year=int(request.form.get("year"))
    car_model=request.form.get("car_models")
    fuel=request.form.get("fuel_type")
    kms_driven=int(request.form.get("kilo_driven"))

    prediction = model.predict(pd.DataFrame([[car_model, kms_driven, fuel, year, company]],
                                            columns=["Name", "Kms_driven", "Fuel_type", "Year", "Company"]))
    print(prediction)
    return str(np.round(prediction[0], 2))
if __name__=="__main__":
    app.run(debug=True)