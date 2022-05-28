import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import missingno as msno
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.info()
df.shape
df.describe()
features = df.columns
cols = (df[features] == 0).sum()
print(cols)
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.isnull().sum()
msno.matrix(df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']])
#Replace the null values with the median of that column:

df['Glucose'].fillna(df['Glucose'].median(), inplace =True)

df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace =True)

df['BMI'].fillna(df['BMI'].median(), inplace =True)
by_Glucose_Age_Insulin_Grp = df.groupby(['Glucose'])

def fill_Insulin(series):
    return series.fillna(series.median())
df['Insulin'] = by_Glucose_Age_Insulin_Grp['Insulin'].transform(fill_Insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
by_BMI_Insulin = df.groupby(['BMI'])

def fill_Skinthickness(series):
    return series.fillna(series.mean())
df['SkinThickness'] = by_BMI_Insulin['SkinThickness'].transform(fill_Skinthickness)
df['SkinThickness'].fillna(df['SkinThickness'].mean(),inplace= True)
df.isnull().sum()by_BMI_Insulin = df.groupby(['BMI'])

import matplotlib.style as style
style.available

style.use('seaborn-pastel')
labels = ["Healthy", "Diabetic"]
df['Outcome'].value_counts().plot(kind='pie',labels=labels, subplots=True,autopct='%1.0f%%', labeldistance=1.2, figsize=(9,9))
from matplotlib.pyplot import figure, show

figure(figsize=(8,6))
ax = sns.countplot(x=df['Outcome'], data=df,palette="husl")
ax.set_xticklabels(["Healthy","Diabetic"])
healthy, diabetics = df['Outcome'].value_counts().values
print("Samples of diabetic people: ", diabetics)
print("Samples of healthy people: ", healthy)
plt.figure()
ax = sns.distplot(df['Pregnancies'][df.Outcome == 1], color ="darkturquoise", rug = True)
sns.distplot(df['Pregnancies'][df.Outcome == 0], color ="lightcoral",rug = True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.figure()
ax = sns.distplot(df['Glucose'][df.Outcome == 1], color ="darkturquoise", rug = True)
sns.distplot(df['Glucose'][df.Outcome == 0], color ="lightcoral", rug = True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.figure()
ax = sns.distplot(df['BloodPressure'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['BloodPressure'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.figure()
ax = sns.distplot(df['SkinThickness'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['SkinThickness'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.figure()
ax = sns.distplot(df['Insulin'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['Insulin'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.figure()
ax = sns.distplot(df['BMI'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['BMI'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.figure()
ax = sns.distplot(df['DiabetesPedigreeFunction'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['DiabetesPedigreeFunction'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.figure()
ax = sns.distplot(df['Age'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['Age'][df.Outcome == 0], color ="lightcoral", rug=True)
sns.distplot(df['Age'], color ="green", rug=True)
plt.legend(['Diabetes', 'No Diabetes', 'all'])
plt.figure(dpi = 120,figsize= (5,4))
mask = np.triu(np.ones_like(df.corr(),dtype = bool))
sns.heatmap(df.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show()
sns.pairplot(df, hue="Outcome",palette="husl")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)
print("Number transactions x_train dataset: ", x_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions x_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,auc
from sklearn.svm import SVC

model=SVC(kernel='rbf')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

from tkinter import *
from PIL import ImageTk,Image
root=Tk()
root.title("Fit&Healthy")
root.iconbitmap("C:/Users/Admin/Desktop")
img=ImageTk.PhotoImage(Image.open("doctor.jpg"))
my_label=Label(root,image=img)
my_label.place(x=0,y=0,relwidth=1,relheight=1)
def button_gluc():
    global a 
    a=float(e.get())
    if a <= 70:  
       label_gluc = Label(root, text= "Low blood sugar level ")
       label_gluc.grid(row=2,column=1)
    if a>=160:
        label_gluc1 = Label(root, text="high blood sugar level ")
        label_gluc1.grid(row=2,column=1)
    if 70<a<160:
        label_gluc2 = Label(root, text= "normal blood sugar level ")
        label_gluc2.grid(row=2,column=1)
        
def button_oxi():
    global b 
    b=float(f.get())
    if b <= 95:  
       label_gluc = Label(root, text= "Low oxygen level ")
       label_gluc.grid(row=6,column=1)
    if b > 95:  
       label_gluc = Label(root, text= "normal oxygen level ")
       label_gluc.grid(row=6,column=1)
    
def button_press():
     global c 
     c=float(g.get())
     if c<= 60:  
       label_gluc = Label(root, text= "Low blood pressure level ")
       label_gluc.grid(row=2,column=1)
     if c>=120:
        label_gluc1 = Label(root, text= "high blood pressure level ")
        label_gluc1.grid(row=10,column=1)
     if 60<c<120:
        label_gluc2 = Label(root, text= "normal blood pressure level ")
        label_gluc2.grid(row=10,column=1)
    
    
def button_exit():
    return
button1=Button(root,text="Glucose",padx=40,pady=20,command=button_gluc)
button2=Button(root,text="oxitext",padx=40,pady=20,command=button_oxi)
button3=Button(root,text="BP",padx=40,pady=20,command=button_press)
button4=Button(root,text="exit",padx=40,pady=20,command=root.destroy)



button1.grid(row=1,column=0)
button2.grid(row=4,column=0)               
button3.grid(row=8,column=0)
button4.grid(row=12,column=0)

e=Entry(root,width=35,borderwidth=5)
e.grid(row=1,column=1,columnspan=3,padx=10,pady=10)
f=Entry(root,width=35,borderwidth=5)
f.grid(row=4,column=1,columnspan=3,padx=10,pady=10)
g=Entry(root,width=35,borderwidth=5)
g.grid(row=8,column=1,columnspan=3,padx=10,pady=10)


               
root.mainloop()
