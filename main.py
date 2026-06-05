from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Fill missing values

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

print("Missing Values After Cleaning:")
print(df.isnull().sum())
# Convert categorical columns into numerical values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
df["Dependents"] = le.fit_transform(df["Dependents"])
df["Education"] = le.fit_transform(df["Education"])
df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

print("\nEncoded Dataset:")
print(df.head())
# Features and Target

X = df.drop(["Loan_ID", "Loan_Status"], axis=1)

y = df["Loan_Status"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_predictions = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:")
print(lr_accuracy)
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, lr_predictions))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, lr_predictions)

print("\nConfusion Matrix:")
print(cm)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)
from sklearn.metrics import roc_auc_score

roc_score = roc_auc_score(y_test, rf_predictions)

print("\nROC-AUC Score:")
print(roc_score)
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved successfully.")
models = ["Logistic Regression", "Random Forest"]
accuracies = [lr_accuracy, rf_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("accuracy_comparison.png")
plt.close()
print("Accuracy comparison graph saved successfully.")
