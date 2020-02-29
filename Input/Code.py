#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:03:17 2020

@author: alexislacabane
"""

import warnings; warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import auc,confusion_matrix,roc_auc_score,roc_curve,precision_score,recall_score,f1_score,accuracy_score

pd.options.display.max_columns = None

df = pd.read_csv('/Users/alexislacabane/Documents/GitHub/Projects/Employee_Attrition/Employee_Attrition.csv', sep=',')

# Change index
df = df.set_index('EmployeeNumber', drop=True)

# Drop repetitive or useless columns
df.drop(['EmployeeCount','PerformanceRating', 'StandardHours', 'DailyRate','MonthlyRate', 
         'EducationField', 'JobRole', 'MonthlyIncome', 'Over18','RelationshipSatisfaction'],axis=1, inplace=True)

# Categorical columns to numerical ones
df.Attrition = df.Attrition.map({'Yes':1,'No':0})
df.OverTime = df.OverTime.map({'Yes':1,'No':0})
df.BusinessTravel = df.BusinessTravel.map({'Travel_Rarely':1,'Travel_Frequently':2, 'Non-Travel':0})

# Winsorize columns
df.TotalWorkingYears = winsorize(df.TotalWorkingYears,limits=[0, 0.05])
df.TrainingTimesLastYear = winsorize(df.TrainingTimesLastYear,limits=[0.05, 0.2])
df.YearsAtCompany = winsorize(df.YearsAtCompany,limits=[0, 0.08])
df.YearsInCurrentRole = winsorize(df.YearsInCurrentRole,limits=[0, 0.05])
df.YearsSinceLastPromotion = winsorize(df.YearsSinceLastPromotion,limits=[0, 0.08])
df.YearsWithCurrManager = winsorize(df.YearsWithCurrManager,limits=[0, 0.05])

# Remove outliers manually
df.drop(df[df.StockOptionLevel==3].index,axis=0, inplace=True)
df.drop(df[df.NumCompaniesWorked==9].index,axis=0, inplace=True)

# Create dummies and reset index
df = pd.get_dummies(data=df,columns=['Gender','MaritalStatus','Department'], drop_first=True)
df = df.reset_index()

# Modify Age to bins
df['Age_bins']=np.where(df.Age < 30, 0, (np.where(df.Age < 40, 1, (np.where(df.Age < 50, 2, 3)))))
df.drop('Age',axis=1,inplace=True)

=============================================DATA VIZ=================================================================

# Variables summary
summary = (df.drop('Attrition',axis=1).describe().T.reset_index())

summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)

val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]

trace  = go.Table(header = dict(values = summary.columns.tolist(),
                                line = dict(color = ['#506784']),
                                fill = dict(color = ['#119DFF']),
                               ),
                  cells  = dict(values = val_lst,
                                line = dict(color = ['#506784']),
                                fill = dict(color = ["lightgrey",'#F5F8FF'])
                               ),
                  columnwidth = [200,60,100,100,60,60,80,80,80])
layout = go.Layout(dict(title = "Variable Summary"))
figure = go.Figure(data=[trace],layout=layout)
py.iplot(figure)

# Numerical Variables violinplots

sns.set()
fig, axes=plt.subplots(nrows=4, ncols=4, figsize=(15,15))
for idx, feat in enumerate(df.drop(['Attrition', 'EmployeeNumber','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate', 'JobSatisfaction' ,'NumCompaniesWorked' ,'PercentSalaryHike','YearsSinceLastPromotion','MaritalStatus_Married','Gender_Male'],axis=1).columns):
    ax=axes[int(idx/4), idx%4]
    sns.violinplot(x='Attrition',y=feat, data=df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout();

# Categorical Variables violinplots
var=df.groupby(['JobLevel','Attrition']).Attrition.count()
var.unstack().plot(kind='bar',stacked=True,color=['blue','orange'],grid=False,figsize=(10,10))
plt.show()

var=df.groupby(['StockOptionLevel','Attrition']).Attrition.count()
var.unstack().plot(kind='bar',stacked=True,color=['blue','orange'],grid=False,figsize=(10,10))
plt.show()

var=df.groupby(['BusinessTravel','Attrition']).Attrition.count()
var.unstack().plot(kind='bar',stacked=True,color=['blue','orange'],grid=False,figsize=(10,10))
plt.show()

var=df.groupby(['OverTime','Attrition']).Attrition.count()
var.unstack().plot(kind='bar',stacked=True,color=['blue','orange'],grid=False,figsize=(10,10))
plt.show()

var=df.groupby(['Age_bins','Attrition']).Attrition.count()
var.unstack().plot(kind='bar',stacked=True,color=['blue','orange'],grid=False,figsize=(10,10))
plt.show()

var=df.groupby(['WorkLifeBalance','Attrition']).Attrition.count()
var.unstack().plot(kind='bar',stacked=True,color=['blue','orange'],grid=False,figsize=(10,10))
plt.show()

# Test VIF for mutlticolinearity
X = df.drop(['Attrition','EmployeeNumber','YearsAtCompany','PercentSalaryHike','TotalWorkingYears','JobInvolvement','WorkLifeBalance','Department_Research & Development'], axis=1)
pd.Series([VIF(X.values, i) for i in range(X.shape[1])],index=X.columns).sort_values(ascending=False)

# Drop columns for Correlation Matrix
data_model = df.drop(['Attrition','EmployeeNumber','YearsAtCompany','PercentSalaryHike','TotalWorkingYears','JobInvolvement','WorkLifeBalance','Department_Research & Development'], axis=1)


# Correlation matrix for variables
correlation = data_model.corr()
matrix_cols = correlation.columns.tolist()
corr_array  = np.array(correlation)
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = "Viridis",
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )
layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )
data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)

# New Variables summary

summary = (data_model.describe().T.reset_index())

summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)

val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]

trace  = go.Table(header = dict(values = summary.columns.tolist(),
                                line = dict(color = ['#506784']),
                                fill = dict(color = ['#119DFF']),
                               ),
                  cells  = dict(values = val_lst,
                                line = dict(color = ['#506784']),
                                fill = dict(color = ["lightgrey",'#F5F8FF'])
                               ),
                  columnwidth = [200,60,100,100,60,60,80,80,80])
layout = go.Layout(dict(title = "Variable Summary"))
figure = go.Figure(data=[trace],layout=layout)
py.iplot(figure)

=========================================MODELS==============================================================

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data_model,df.Attrition,test_size=1/3,random_state=42)

# Model 1: Logistic Regression without normalizing HourlyRate
model1 = LogisticRegression(class_weight='balanced')
res1 = model1.fit(X_train,y_train)
pred1 = model1.predict(X_test)
conf1 = confusion_matrix(y_test,pred1)
conf1

model1_roc=roc_auc_score(y_test,pred1)
fpr,tpr,thresholds=roc_curve(y_test, model1.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model1 (area={model1_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

# Model 2: Logistic Regression after normalizing HourlyRate

data_modelt = data_model.copy()
data_modelt.HourlyRate = (data_model.HourlyRate-data_model.HourlyRate.min())/(data_model.HourlyRate.max()-data_model.HourlyRate.min())

Xt_train, Xt_test, yt_train, yt_test = train_test_split(data_modelt,df.Attrition,test_size=1/3,random_state=42)

sns.distplot(data_modelt.HourlyRate)

model2 = LogisticRegression(class_weight='balanced')
res2 = model2.fit(Xt_train,yt_train)
pred2 = model2.predict(Xt_test)
conf2 = confusion_matrix(yt_test,pred2)
conf2

model2_roc=roc_auc_score(yt_test,pred2)
fpr,tpr,thresholds=roc_curve(yt_test, model2.predict_proba(Xt_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model2 (area={model2_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

# Model 3 & 4: KNearest Neighbors

model3 = KNeighborsClassifier()
res3 = model3.fit(X_train,y_train)
pred3 = model3.predict(X_test)
conf3 = confusion_matrix(y_test,pred3)
conf3

model3_roc=roc_auc_score(y_test,pred3)
fpr,tpr,thresholds=roc_curve(y_test, model3.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model3 (area={model3_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

model4 = KNeighborsClassifier(n_neighbors=7,weights='distance')
res4 = model4.fit(X_train,y_train)
pred4 = model4.predict(X_test)
conf4 = confusion_matrix(y_test,pred4)
conf4

model4_roc=roc_auc_score(y_test,pred4)
fpr,tpr,thresholds=roc_curve(y_test, model4.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model4 (area={model4_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

# Model 5: Naive Bayes Model

model5 = GaussianNB()
res5 = model5.fit(X_train,y_train)
pred5 = model5.predict(X_test)
conf5 = confusion_matrix(y_test,pred5)
conf5

model5_roc=roc_auc_score(y_test,pred5)
fpr,tpr,thresholds=roc_curve(y_test, model5.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model5 (area={model5_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

# Model 6 & 7: Decision Tree

model6 = DecisionTreeClassifier()
res6 = model6.fit(X_train,y_train)
pred6 = model6.predict(X_test)
conf6 = confusion_matrix(y_test,pred6)
conf6

model6_roc=roc_auc_score(y_test,pred6)
fpr,tpr,thresholds=roc_curve(y_test, model6.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model6 (area={model6_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

model7 = DecisionTreeClassifier(class_weight='balanced')
res7 = model7.fit(X_train,y_train)
pred7 = model7.predict(X_test)
conf7 = confusion_matrix(y_test,pred4)
conf7

model7_roc=roc_auc_score(y_test,pred7)
fpr,tpr,thresholds=roc_curve(y_test, model7.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model7 (area={model7_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

# Model 8 & 9: Random Forest

model8 = RandomForestClassifier()
res8 = model8.fit(X_train,y_train)
pred8 = model8.predict(X_test)
conf8 = confusion_matrix(y_test,pred8)
conf8

model8_roc=roc_auc_score(y_test,pred8)
fpr,tpr,thresholds=roc_curve(y_test, model8.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model8 (area={model8_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

model9 = RandomForestClassifier(class_weight='balanced')
res9 = model9.fit(X_train,y_train)
pred9 = model9.predict(X_test)
conf9 = confusion_matrix(y_test,pred9)
conf9

model9_roc=roc_auc_score(y_test,pred9)
fpr,tpr,thresholds=roc_curve(y_test, model9.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model9 (area={model9_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

# Model 10: Support Vector Machine

model10 = SVC(probability=True)
res10 = model10.fit(X_train,y_train)
pred10 = model10.predict(X_test)
conf10 = confusion_matrix(y_test,pred10)
conf10

model10_roc=roc_auc_score(y_test,pred10)
fpr,tpr,thresholds=roc_curve(y_test, model10.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model10 (area={model10_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

# Model 11: Nu SVC

model11 = NuSVC(nu=0.2,probability=True,gamma='scale')
res11 = model11.fit(X_train,y_train)
pred11 = model11.predict(X_test)
conf11 = confusion_matrix(y_test,pred11)
conf11

model11_roc=roc_auc_score(y_test,pred11)
fpr,tpr,thresholds=roc_curve(y_test, model11.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model11 (area={model11_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()


## Compare models
## OBJECTIVE: MINIMIZE FN (Type II Error)

lst=[]
for i in range(1,13):
    FP = eval(f'conf{i}')[0][1]
    FN = eval(f'conf{i}')[1][0]
    TP = eval(f'conf{i}')[1][1]
    TN = eval(f'conf{i}')[0][0]
    
    FNR = FN/(TP+FN)*100
    FPR = FP/(FP+TN)*100
    ACC = (TP+TN)/(TP+FP+FN+TN)*100
    AUC = eval(f'model{i}_roc')
    
    lst.append([i,FNR,FPR,ACC,AUC])
    
results = pd.DataFrame(lst, columns=['Model','False_negative_rate','False_Positive_rate','Overall_Accuracy','Area_Under_Curve'])
results = results.set_index('Model')

------> # Model 1 is the best (Logistic Regression)


# Model 1 with PCA

pca = PCA(0.85)
pca.fit(data_model)
pca.explained_variance_ratio_
X_train_PCA=pca.transform(X_train)
X_test_PCA=pca.transform(X_test)

model12 = LogisticRegression(class_weight='balanced')
res12 = model12.fit(X_train_PCA,y_train)
pred12 = model12.predict(X_test_PCA)
conf12 = confusion_matrix(y_test,pred12)
conf12

model12_roc=roc_auc_score(y_test,pred12)
fpr,tpr,thresholds=roc_curve(y_test, model12.predict_proba(X_test_PCA)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model1 (area={model12_roc})')
plt.plot([0,1],[0,1])
plt.legend()
plt.show()





