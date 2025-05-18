# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:55:41 2025

@author: tianr
"""

import pandas as pd
import numpy as np
import os


data = pd.read_csv('D:/DTR/MyWork/2025/DataMining/GROUPProject/TEMP/SEER_ori.csv')
data = data.dropna()


#1.数据预处理
#(1)Sex列处理
sex_map = {'Female': 0, 'Male': 1}
data['Sex'] = data['Sex'].map(sex_map)

#(2)Race列处理
data = data[data["Race recode (White, Black, Other)"] != 'Unknown']
race_map = {'White': 0, 'Black': 1, 'Other(American Indian/AK Native, Asian/Pacific Islander)': 2}
data["Race recode (White, Black, Other)"] = data["Race recode (White, Black, Other)"].map(race_map)
data = data.dropna(subset=["Race recode (White, Black, Other)"])  
    
#(3)Age列处理
age_map = {'80-84 years':82, '60-64 years':62, '45-49 years':47, '70-74 years':72,
           '85+ years':85, '55-59 years':57, '75-79 years':77, '65-69 years':67,
           '50-54 years':52, '35-39 years':37, '40-44 years':42, '25-29 years':27,
           '30-34 years':32, '15-19 years':17, '10-14 years':12, '20-24 years':22,
           '01-04 years':2, '05-09 years':7, '00 years':0}
data["Age recode with <1 year olds"] = data["Age recode with <1 year olds"].map(age_map)

#(4)Primary Site & Site 列处理
#从探索性分析的结果可知，这两列对生存时间的影响非常小，且取值空间较大，
#若进行One-Hot编码会导致学习时间上升较大，因此不将其作为用来进行学习的特征。

#(5)Grade列处理
data = data[(data['Grade Recode (thru 2017)'] == 'Well differentiated; Grade I') |
            (data['Grade Recode (thru 2017)'] == 'Moderately differentiated; Grade II') |
            (data['Grade Recode (thru 2017)'] == 'Poorly differentiated; Grade III') |
            (data['Grade Recode (thru 2017)'] == 'Undifferentiated; anaplastic; Grade IV')] 
grade_map = {'Poorly differentiated; Grade III':3,'Moderately differentiated; Grade II':2,
             'Well differentiated; Grade I':1,'Undifferentiated; anaplastic; Grade IV':4}
data["Grade Recode (thru 2017)"] = data["Grade Recode (thru 2017)"].map(grade_map)

#(6)Stage Group 列处理
data.dropna(subset=['Derived AJCC Stage Group, 7th ed (2010-2015)'], inplace=True)
data = data[(data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'Blank(s)') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'UNK Stage') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IEA') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'OCCULT') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'INOS') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIINOS') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIEA') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IE') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIEB') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IINOS') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIIESB') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIIEA') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IVNOS') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'ISA') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIE') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IBNOS') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIISA') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIIE') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IEB') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IISA') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIIEB') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIESB') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIISB') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIIESA') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIESA') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IS') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIS') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIANOS') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IANOS') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'ISB') &
            (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIES') & (data['Derived AJCC Stage Group, 7th ed (2010-2015)'] != 'IIIES')] 
stage_map = {'0': 1, '0a': 1, '0is': 1, 'I': 2, 'IA': 3, 'IA1': 3, 'IA2': 3, 'IB': 4, 'IB1': 4, 'IB2': 4, 'IC': 5,
             'II': 6, 'IIA': 7, 'IIA1': 7, 'IIA2': 7, 'IIB': 8, 'IIC': 9,
             'III': 10, 'IIIA': 11, 'IIIB': 12, 'IIIC': 13, 'IIIC1': 13, 'IIIC2': 13,
             'IV': 14, 'IVA': 15, 'IVB': 16, 'IVC': 17}
data["Derived AJCC Stage Group, 7th ed (2010-2015)"] = data["Derived AJCC Stage Group, 7th ed (2010-2015)"].map(stage_map)

#(7)Tumor Size 列处理
data = data[(data['CS tumor size (2004-2015)'] != 'Blank(s)')]
data['CS tumor size (2004-2015)'] = data['CS tumor size (2004-2015)'].astype(int)
data = data[(data['CS tumor size (2004-2015)'] <= 600)]

#(8)mets at dx 列处理
data.dropna(subset=['CS mets at dx (2004-2015)'], inplace=True)
data = data[(data['CS mets at dx (2004-2015)'] != 99)]

unique_sorted = sorted(set(data['CS mets at dx (2004-2015)'].unique()))
print(unique_sorted)


#(9)Surg Prim Site 列处理
data.dropna(subset=['CS mets at dx (2004-2015)'], inplace=True)
data = data[(data['RX Summ--Surg Prim Site (1998+)'] != 99)]


unique_sorted = sorted(set(data['RX Summ--Surg Prim Site (1998+)'].unique()))
print(unique_sorted)



#(10)Survival months 列处理
data = data[(data['Survival months'] != 'Unknown')]
data['Survival months'] = data['Survival months'].astype(int)


#2.机器学习模型构建
#(1)数据集划分，利用Scikit-learn库的train_test_split函数将数据集以8：2的比例进行划分。
from sklearn.model_selection import train_test_split
Selected_features=['Sex','Race recode (White, Black, Other)','Age recode with <1 year olds',
                   'Grade Recode (thru 2017)','Derived AJCC Stage Group, 7th ed (2010-2015)',
                   'CS tumor size (2004-2015)','CS mets at dx (2004-2015)','RX Summ--Surg Prim Site (1998+)']
X=data[Selected_features]
Y=data['Survival months']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)#以8：2的比例划分训练集与测试集

#(2)利用逻辑回归模型训练多分类模型
from sklearn.metrics import accuracy_score, classification_report#调用ACC系数，和分类指标函数
from sklearn.linear_model import LogisticRegression#调用逻辑回归模型
model = LogisticRegression()#实例化逻辑回归模型
model.fit(X_train, Y_train)#输入训练集，训练逻辑回归模型
predictions = model.predict(X_test)#在测试集上进行预测
 
print(classification_report(Y_test, predictions))#输出精度指标表
print('Predicted labels: ', predictions)#输出预测结果
print('Accuracy: ', accuracy_score(Y_test, predictions))#输出ACC系数值


#3.“腌制”模型
import pickle #调用“腌制”库
model_filename = 'seerpredict-model.pkl'#设定文件名
pickle.dump(model, open(model_filename,'wb'))#对模型进行“腌制”
 
model = pickle.load(open('seerpredict-model.pkl','rb'))#加载“腌制”好的模型
#测试模型，其中参数分别是出现的'Sex','Race recode (White, Black, Other)','Age recode with <1 year olds',
#'Grade Recode (thru 2017)','Derived AJCC Stage Group, 7th ed (2010-2015)',
#'CS tumor size (2004-2015)','CS mets at dx (2004-2015)','RX Summ--Surg Prim Site (1998+)'
print(model.predict([[0,0,0,1,1,80,40,40]]))


































