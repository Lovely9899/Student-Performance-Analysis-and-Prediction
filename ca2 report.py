# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:24:06 2025

@author: hp
"""
import dtale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler


df=pd.read_excel("C:/Users/hp/Downloads/Students_Performance_data_set.xlsx")

# Convert CGPA to numeric
df['CGPA'] = pd.to_numeric(df['What is your current CGPA?'])
df.info()
df.describe()
we=dtale.show(df)
we.open_browser()
df.isnull().sum()
print(df['Pass'].value_counts())
x=df['What are the skills do you have ?'].mode()[0] 
df['What are the skills do you have ?'].fillna(x,inplace=True) 
#statstical analysis
print("Mean CGPA:", df['CGPA'].mean())
print("Std Deviation:", df['CGPA'].std())
print("Skewness:", df['CGPA'].skew())
df.groupby('Gender')['CGPA'].describe()


                                   
# Created Pass/Fail column for logistic regression
df['Pass'] = (df['CGPA'] >= 2.5).astype(int)
df['Gender']=df['Gender'].map({'Male':1,'Female':0})

#plotting

#histogram
plt.figure(figsize=(8,5))
sns.histplot(df['CGPA'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Count')
plt.show()

#box plot
plt.figure(figsize=(7,5))
sns.boxplot(x='Gender', y='CGPA', data=df, palette='pastel')
plt.title('CGPA by Gender')
plt.show()

# bar plot
plt.figure(figsize=(10,6))
sns.barplot(x='How many hour do you study daily?', y='CGPA',hue='Gender', data=df)
plt.title("CGPA vs Study Hours")
plt.xlabel("Daily Study Hours")
plt.ylabel("CGPA")
plt.show()

#pie chart
top_skills = df['What are the skills do you have ?'].value_counts().head(5)

plt.figure(figsize=(6,6))
plt.pie(top_skills, labels=top_skills.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 5 Reported Skills")
plt.axis('equal')
plt.show()





#features selection
selected_features = ['How many hour do you study daily?',
    'How many times do you seat for study in a day?',
    'Do you have any health issues?']
     

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)  # Convert everything to string
    df[col] = LabelEncoder().fit_transform(df[col])


# Split into features and targets
X = df[selected_features]
y_reg = df['CGPA']


# Train/test split
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2)



# LINEAR REGRESSION

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)
y_reg_pred = lin_reg.predict(X_test)

print(" Linear Regression Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_reg_test, y_reg_pred))
print("R² Score:", r2_score(y_reg_test, y_reg_pred))


#plotting
plt.scatter(X, y_reg, color='blue')
plt.plot(X, y_reg_pred, color='red', label='Prediction')
plt.xlabel("Study Hours")
plt.ylabel("CGPA")
plt.title("Study Hours vs CGPA")
plt.legend()
plt.show()


# LOGISTIC REGRESSION
selected_features = [
'Gender',
 'Age',
 'How many hour do you study daily?',
 'How many times do you seat for study in a day?',
 'What is your monthly family income?']

X = df[selected_features]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
log_reg = LogisticRegression()


#log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print("\n🔶 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))












