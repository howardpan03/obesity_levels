資料是取自https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels
參考https://www.mdpi.com/2076-3417/13/6/3875 做資料處理  
__資料資訊:__  
資料集包括根據墨西哥、秘魯和哥倫比亞國家個人的飲食習慣和身體狀況估計其肥胖程度的資料。77%的數據是使用Weka工具和SMOTE篩選綜合產生的，23%的數據是透過網路平台直接從用戶收集的。  
1.Age :請問你的年齡?(特徵，數值型態)  
2.Gender :請問你的性別?(特徵，二元分類)  
3.Height :請問你的身高?(特徵，數值型態)  
4.Weight :請問你的體重?(特徵，數值型態)  
5.CALC :請問你經常喝酒?(特徵，分類)  
6.FAVC :請問你經常吃高熱量的食物嗎?(特徵，二元分類)  
7.FCVC :請問你吃飯時經常會搭配蔬菜嗎?(特徵，分類)  
8.NCP :請問你一天吃幾頓正餐?(特徵，分類)  
9.SCC :請問你會監測你一天吃多少卡路里嗎?(特徵，二元分類)  
10.SMOKE :請問你抽菸嗎?(特徵，二元分類)  
11.CH2O :請問你一天喝多少水?(特徵，分類)  
12.family_history_with_overweight :請問你的家庭成員中是否有人超重?(特徵，二元分類)  
13.FAF :請問你一個禮拜運動幾天?(特徵，分類)  
14.TUE :請問你一天花多少時間使用3C產品?(特徵，分類)  
15.CAEC :請問你經常在兩餐之間吃任何食物嗎?(特徵，分類)  
16.MTRANS :請問你通常使用哪種交通工具?(特徵，分類)  
17.NObeyesdad :肥胖等級(目標，分類)  
__其他說明:__  
code裡的obesity_levels資料夾是django的資料，且Obesity_Levels.py執行後會有obesity_level_ml_model.sav，我已直接放在code\obesity_levels裡，方便django使用
