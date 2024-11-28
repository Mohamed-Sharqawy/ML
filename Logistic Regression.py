#modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("framingham.csv")
df.drop(columns = 'education', inplace = True)
df = df[df['cigsPerDay'].notna()]
df = df[df['heartRate'].notna()]
df = df[df['BMI'].notna()]
df = df[df['totChol'].notna()]
df.fillna({'BPMeds': 0}, inplace=True)
mean = round(df['glucose'].mean())
df.fillna({'glucose': mean}, inplace = True)

# initialize 
y = np.array(df['TenYearCHD'])
df.drop(columns = 'TenYearCHD', inplace = True)
x = np.array(df)
x_scaled = StandardScaler().fit_transform(x)

# Splitting the Data 
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.25)

# create the model 
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred) * 100

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels =['0','1'])

plt.figure(figsize = (8, 6))
disp.plot(cmap = plt.cm.Reds)
plt.title("Confusion Matrix")