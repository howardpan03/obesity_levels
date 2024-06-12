from django.shortcuts import render
# Create your views here.

def home(request):    
    return render(request, 'home.html')

#產生預測
def getPredictions(Age,Gender,Height,Weight,BMI,CALC,FAVC,FCVC,NCP,
                   SCC,SMOKE,CH2O,FHWO,FAF,TUE,CAEC,MTRANS):
    import pickle
    #讀取pickle儲存過的模型
    model = pickle.load(open("obesity_level_ml_model.sav", "rb"))
    #進行預測
    prediction = model.predict([[Age,Gender,Height,Weight,BMI,CALC,FAVC,FCVC,NCP,
                                 SCC,SMOKE,CH2O,FHWO,FAF,TUE,CAEC,MTRANS]])
    
    if prediction ==0:
        return "Insufficient_Weight(體重不足)"
    elif prediction ==1:
        return "Normal_Weight(正常體重)"
    elif prediction ==2:
        return "Overweight(超重)"
    elif prediction ==3:
        return "Obesity_Type_I(肥胖I型)"
    elif prediction ==4:
        return "Obesity_Type_II(肥胖II型)"
    elif prediction ==5:
        return "Obesity_Type_III(肥胖III型)"
    else:
        return "error"

def result(request):
    Age = int(request.GET['Age'])
    Gender = int(request.GET['Gender'])
    Height = float(request.GET['Height'])
    Weight = float(request.GET['Weight'])
    BMI = round(Weight/((Height/100)**2),2)
    CALC = int(request.GET['CALC'])
    FAVC = int(request.GET['FAVC'])
    FCVC = int(request.GET['FCVC'])
    NCP = int(request.GET['NCP'])
    SCC = int(request.GET['SCC'])
    SMOKE = int(request.GET['SMOKE'])
    CH2O = int(request.GET['CH2O'])
    FHWO = int(request.GET['FHWO'])
    FAF = int(request.GET['FAF'])
    TUE = int(request.GET['TUE'])
    CAEC = int(request.GET['CAEC'])
    MTRANS = int(request.GET['MTRANS'])

    #取得資料後帶給上方的自訂函數(getPredictions)取得預測結果
    #再把預測結果帶回給result
    result = getPredictions(Age,Gender,Height,Weight,BMI,CALC,FAVC,FCVC,
                            NCP,SCC,SMOKE,CH2O,FHWO,FAF,TUE,CAEC,MTRANS)

    #在result.html中可以顯示result和BMI
    return render(request, 'result.html', {'result':result,'BMI':BMI})