# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')       #忽略警示語
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def ml_model(x):
    #LinearRegression
    lm = LinearRegression()
    lm.fit(x, y)
    print('Model: ' + 'LinearRegression')
    print('score: %.4f'%lm.score(x, y))
    print('----------')
    
    #LogisticRegression
    #由於LogisticRegression主要是做二元分類，但此資料是多元分類，所以不使用
    
    #Tree
    tree_name = ['RandomForestClassifier','LGBMClassifier',
                 'DecisionTreeClassifier','ExtraTreesClassifier']
    
    #LGBMClassifier(verbose=-1)
    #列印訊息的詳細程度。
    #如果 verbosity 小於 0，則列印訊息僅顯示嚴重錯誤。
    #如果將 verbosity 設定為 0，則列印訊息會包含錯誤與警告。
    #如果 verbosity 是 1，則列印訊息會顯示詳細資訊。verbosity 大於 1 顯示列印訊息中最多的資訊，可用於偵錯。
    
    # tree_models = [RandomForestClassifier(),LGBMClassifier(verbose=-1),
    #                 DecisionTreeClassifier(),ExtraTreesClassifier()]
    
    tree_models = [RandomForestClassifier(random_state=30),LGBMClassifier(verbose=-1),
                    DecisionTreeClassifier(),ExtraTreesClassifier(random_state=30)]
    
    #一般會設定test_size=0.2~test_size=0.25(預設值)，因此選擇test_size=0.25
    #random_state則隨便選
    for name, model in zip(tree_name,tree_models):
        XTrain,XTest,yTrain,yTest = train_test_split(x,y,test_size=0.25, random_state=30)
        Model = model
        Model.fit(XTrain, yTrain)
        pred = Model.predict(XTest)
        accuracy = accuracy_score(yTest, pred)
        print('Model: %s'%name)
        print('Accuracy_score: %.4f'%accuracy)
        print('----------')
        
    #KNN
    #因此通常建議k值須小於樣本數的平方根較好。(樣本數為2111，平方根為45)
    #一般會設定test_size=0.2~test_size=0.25(預設值)，因此選擇test_size=0.2
    #random_state則隨便選
    XTrain,XTest,yTrain,yTest = train_test_split(x,y,test_size=0.25, random_state=30)
    KNN = KNeighborsClassifier(n_neighbors=45)
    KNN.fit(XTrain, yTrain)
    pred = KNN.predict(XTest)
    accuracy = accuracy_score(yTest, pred)
    print('Model: ' + 'KNN')
    print('Accuracy_score: %.4f'%accuracy)
    print('----------')
    
    #DBSCAN
    #由於DBSCAN的功能是用做分群，無法看準確度，因此我就不拿來做訓練

#匯入csv檔
df = pd.read_csv('Obesity_Levels.csv')

#顯示資料資訊
#df.info()

#把Age、FCVC、NCP、CH2O、FAF、TUE轉整數，
#Age基本上都是講整數，除了嬰兒會特別講幾個月大，所以我還是把所有Age轉成整數
#FCVC、CH2O、FAF、TUE是分類，所以轉成整數
#NCP是1天吃幾頓，所以轉成整數
columns = ['Age','FCVC','NCP','CH2O','FAF','TUE']
for column in columns:
    df[column] = df[column].astype(int)

#把Height從公尺轉公分
np_height = np.array(df['Height'])
np_height_cm = np_height*100

#Height取到小數2位
np_height_cm = np.around(np_height_cm,2)

#Weight取到小數2位
np_weight = np.array(df['Weight'])
np_weight_kg = np.around(np_weight,2)

#新增BMI欄位
np_bmi = np_weight_kg / (np_height ** 2)
np_bmi = np.round(np_bmi,2)
df.insert(4,column='BMI',value=np_bmi)

#把修改後的資料帶回DataFrame
df['Height'] = pd.DataFrame(np_height_cm)
df['Weight'] = pd.DataFrame(np_weight_kg)

#有的BMI被分錯類別，所以重新分類
df.loc[40<=df['BMI'],'NObeyesdad'] = 'Obesity_Type_III'
df.loc[df['BMI']<40,'NObeyesdad'] = 'Obesity_Type_II'
df.loc[df['BMI']<35,'NObeyesdad'] = 'Obesity_Type_I'
df.loc[df['BMI']<30,'NObeyesdad'] = 'Overweight'
df.loc[df['BMI']<25,'NObeyesdad'] = 'Normal_Weight'
df.loc[df['BMI']<18.5,'NObeyesdad'] = 'Insufficient_Weight'
    
#把字串資料轉數字
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['CALC'] = le.fit_transform(df['CALC'])
df['FAVC'] = le.fit_transform(df['FAVC'])
df['SCC'] = le.fit_transform(df['SCC'])
df['SMOKE'] = le.fit_transform(df['SMOKE'])
df['family_history_with_overweight'] = le.fit_transform(df['family_history_with_overweight'])
df['CAEC'] = le.fit_transform(df['CAEC'])
df['MTRANS'] = le.fit_transform(df['MTRANS'])
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])

#欄位名稱過長，所以改名
df= df.rename(columns={'family_history_with_overweight':'FHWO'})

#顯示資料資訊
#df.info()

#取自變數X、應變數y
X = df.iloc[:,0:17]
y = df['NObeyesdad']

#自變數X進行標準化
sclar = StandardScaler()
X_std = sclar.fit_transform(X)

#計算相關矩陣，來進行特徵篩選
correlation_matrix = df.corr()  #型態為DataFrame

plt.rcParams['font.family'] = 'Microsoft JhengHei'#設定中文
plt.rcParams['font.size'] = 18    #設定字體大小  
plt.rcParams['axes.unicode_minus'] = False  #讓正負號能正常顯示

#繪製相關矩陣熱圖
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='BuGn', fmt=".2f")
#sns.heatmap的參數依序為(數據，添加圖片內的註釋，設定顏色，註釋使用的格式)
plt.title('相關矩陣熱圖', weight='bold', size=30)
plt.savefig('correlation_heatmap.jpg')
plt.show()

#特徵篩選
#少了'SMOKE'因為它跟'NObeyesdad'的相關係數取絕對值最小
new_X = X[['Age','Gender','Height','Weight','BMI','CALC','FAVC','FCVC','NCP',
           'SCC','CH2O','FHWO','FAF','TUE','CAEC','MTRANS']]

len_set_y  = len(set(y))    #應變數y分成幾類，方便我帶入KMeans的n_clusters參數

print() #單純換行，讓下方的print('未標準化')輸出結果不被繪圖的警語改變排版
print('未標準化')
ml_model(X)
kmeans = KMeans(n_clusters=len_set_y, n_init='auto', random_state=30)
kmeans.fit(X)
labels = kmeans.labels_
#print('KMeans標籤未修正的準確率: %.4f'%accuracy_score(y, labels))
#KNN的分群結果。因為資料量太大，如果直接輸出結果，其他的輸出結果不會被顯示
#因此建立DataFrame方便進行labels_修正
df_labels = pd.DataFrame([labels,y],index=['no_fix','real']).T
pred_y = np.choose(labels, [3,0,5,5,0,4])
#1,4:0
#2,3,4:1
#0,2,3,5:2
#0,5:3
#0,5:4
#2,3,4:5
accuracy = accuracy_score(y, pred_y)
#冒號左邊是被取代的，冒號右邊是要取代的，方便我來測試提高準確率
print('Model: ' + 'KMeans(標籤已修正)')
print('Accuracy_score: %.4f'%accuracy)

print('--------------------分割線--------------------')

print('標準化')
ml_model(X_std)
kmeans = KMeans(n_clusters=len_set_y, n_init='auto', random_state=30)
kmeans.fit(X_std)
std_labels = kmeans.labels_
#print('KMeans標籤未修正的準確率: %.4f'%accuracy_score(y, std_labels))
#KNN的分群結果。因為資料量太大，如果直接輸出結果，其他的輸出結果不會被顯示
#因此建立DataFrame方便進行labels_修正
df_std_labels = pd.DataFrame([std_labels,y],index=['no_fix','real']).T
pred_y = np.choose(std_labels, [4,5,0,5,1,5])
#1,2,3,4,5:0
#1,2,3,4,5:1
#0,1,3,5:2
#0,1,3,5:3
#0:4
#1,3,5:5
accuracy = accuracy_score(y, pred_y)
#冒號左邊是被取代的，冒號右邊是要取代的，方便我來測試提高準確率
print('Model: ' + 'KMeans(標籤已修正)')
print('Accuracy_score: %.4f'%accuracy)

print('--------------------分割線--------------------')

print('新的X未標準化')        #以新的X來提升準確率
ml_model(new_X)
#因為KMeans的準確率很小，因此就不再訓練KMeans

#利用pickle儲存下方模型
XTrain,XTest,yTrain,yTest = train_test_split(X,y,test_size=0.25, random_state=30)
lgbm = LGBMClassifier(verbose=-1)
lgbm.fit(XTrain, yTrain)
#sav檔案寫入(python資料格式儲存成sav檔案)
pickle.dump(lgbm,open('obesity_level_ml_model.sav','wb'))