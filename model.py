import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv("iris.csv")
x=df[["SepalLengthCm", "SepalWidthCm",
        "PetalLengthCm", "PetalWidthCm"]]
y=df["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
classification_rep=classification_report(y_test,y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("enter the values of sepal length,sepal width,petal length,petal width")
sepal_length=float(input("Enter sepal length: "))
sepal_width=float(input("Enter sepal width: "))
petal_length=float(input("Enter petal length: "))
petal_width=float(input("Enter petal width: "))
new_data=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns=x.columns)
prediction=model.predict(new_data)
print("Prediction:", prediction)
