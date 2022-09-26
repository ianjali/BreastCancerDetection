
import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import RobustScaler

app= Flask(__name__,template_folder='templates/')
model=pickle.load(open('model.pkl','rb'))
pca= pickle.load(open('pca.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def main_page():
    return render_template('index.html')
@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method =="GET":
        return render_template('form.html')
    colums = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean',\
            'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',\
            'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',\
            'concave points_se','symmetry_se','fractal_dimension_se',\
            'radius_worst','texture_worst', 'perimeter_worst','area_worst','smoothness_worst',\
            'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
    list =[]
    for j in colums:
        list.append(float(request.form[j]))
    #listM= [20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902]
    #listB =[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
    #list = listB
    array_list = [np.array(list)]
    array_features = array_list
    array_features = scaler.transform(array_features)
    X_PCA_t = pca.transform(array_features)
    y_s = model.predict_proba(X_PCA_t)#[:,1] >0.5
    prediction = y_s[:,1] > 0.5 #model threshhold
    print(f'prediction is {prediction}')
    percentage = int(y_s[:,1] * 100)
    if ~prediction:
        return render_template("form.html",result = 'Congratulations! You are free from breast cancer ')
    else:
        return render_template("form.html",result1 = f"Glad you took the test, there is a {percentage}% chance you have breast cancer. We are in this together.")
if __name__=='__main__':
    app.run(debug=True)
