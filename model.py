import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

#loading the dataset
data = pd.read_csv("dataset/disease_data.csv")
# print(data.head())

#split the data into features and labels
x = data[['Fever', 'Cough', 'Headache', 'Fatigue']]
y = data['Disease']

#training
x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#create and train data
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

joblib.dump(model, 'disease_predictor_model.joblib')
print("model trained and saved successfullly")