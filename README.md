<H3>ENTER YOUR NAME: RAJESH A</H3>
<H3>ENTER YOUR REGISTER NO : 212222100042</H3>
<H3>EX. NO.6</H3>
<H3>DATE:27/04/2024</H3>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate 

## Program:
```py
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
data = pd.read_csv('heart.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
training_loss = mlp.fit(X_train, y_train).loss_curve_
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss Values")
plt.show()
conf_matrix=confusion_matrix(y_test,y_pred)
classification_rep=classification_report(y_test,y_pred)
print("Confusion Matrix")
print(conf_matrix)
print("Classification Report")
print(classification_rep)
```

<H3>Output:</H3>

## Accuracy

![r](https://github.com/Rajeshanbu/EX-6-NN/assets/118924713/ddc9e883-86da-4cc0-9a0e-8b804f40335c)

## Iteration vs Loss

![rr](https://github.com/Rajeshanbu/EX-6-NN/assets/118924713/bd268e57-949a-4c29-b14a-90edd0e4a1e8)


## Confusion Matrix

![rrr](https://github.com/Rajeshanbu/EX-6-NN/assets/118924713/18509f85-524b-4b06-8734-179eb6b2022a)


## Classification Report

![rrrr](https://github.com/Rajeshanbu/EX-6-NN/assets/118924713/5ba2fa87-6c9c-4998-b021-623f6f4d7595)



<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
