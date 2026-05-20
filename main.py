import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

print(data.head())

# Input features
X = data[['StudyHours', 'Attendance', 'SleepHours', 'PreviousMarks']]

# Output
y = data['Result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy * 100)

# Custom student data
new_data = pd.DataFrame(
    [[6, 80, 7, 75]],
    columns=['StudyHours', 'Attendance', 'SleepHours', 'PreviousMarks']
)

result = model.predict(new_data)

if result[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")