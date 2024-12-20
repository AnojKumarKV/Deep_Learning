"""
This is a Churn Modelling ANN Project to predict whether the customers of a bank will stay or leave.
The .csv file has data of customers for a period of 6 months. Left = '1', Stayed = '0'
"""

# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# PART 1 - DATA PREPROCESSING
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values #selected the independent columns
y = dataset.iloc[:, -1].values #selected the dependent column
# Label Encoding the 'Gender' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
# One Hot Encoding the 'Geography' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Splitting the dataset into Training and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PART 2 - BUILDING THE ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding the input, first and the second hidden layers
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# PART 3 - TRAINING THE ANN
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Training the ANN
ann.fit(X_train, y_train, batch_size = 32, epochs = 100) #run the test: accuracy is 0.8638 over 100 epochs

"""
# PART 4 - PREDICTING THE RESULT FOR A SINGLE OBSERVATION
Use our ANN model to predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) - the customer will stay!
"""

# PART 5 - PREDICTIONS
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))



