
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.externals import joblib

reloadModel = joblib.load('./models/model.pkl')




# Create your views here.

def homepage(request):
    return render(request,'home.html') 

def prediction(request):
    temp = {}
    temp['Glucose'] = request.POST.get('Glucose')
    temp['Insulin'] = request.POST.get('Insulin')
    temp['BMI'] = request.POST.get('BMI')
    temp['Age'] = request.POST.get('Age')
    testdata = pd.DataFrame({'x' :temp}).transpose()
    pred = reloadModel.predict(testdata)
    if pred == [1.]:
        pred = 'Diabetic'
    else:
        pred = 'Not Diabetic'
   
    context = {'pred': pred}
    
    return render(request,'prediction.html',context)  
   
