import pandas as pd


data = pd.read_csv('data.csv')

# data preprocessing
# deal with missing values
data.fillna(data.median(), inplace=True)
for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# encode
data = pd.get_dummies(data, drop_first=True)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['category_feature'] = label_encoder.fit_transform(data['category_feature'])

# normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)


# model choice and training
# user churn or not can be a binary problem
# Logistic Regression, Random Forest, XGBoost, SVM, NN

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = data_scaled.drop('Churn', axis=1)
y = data_scaled['Chrun']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# model tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_sample_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# deploy
import joblib

joblib.dump(model, 'churn_model.pkl')
loaded_model = joblib.load('churn_model.pkl')
predictions = loaded_model.predict(X_test)
