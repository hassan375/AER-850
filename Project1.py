import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
import joblib

# Step 1
# ----------------------------------------------------------------------------
# Load the CSV file into a dataframe
df = pd.read_csv('Project_1_Data.csv')

# Display the first few rows of the dataframe
print(df.head())


# Step 2
# ----------------------------------------------------------------------------
print(df.describe())
# Plot histograms for X, Y, and Z coordinates to understand their distributions
plt.figure(figsize=(12, 4))

# Plot for X coordinate
plt.subplot(1, 3, 1)
plt.hist(df['X'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of X coordinate')

# Plot for Y coordinate
plt.subplot(1, 3, 2)
plt.hist(df['Y'], bins=20, color='green', edgecolor='black')
plt.title('Distribution of Y coordinate')

# Plot for Z coordinate
plt.subplot(1, 3, 3)
plt.hist(df['Z'], bins=20, color='red', edgecolor='black')
plt.title('Distribution of Z coordinate')

plt.tight_layout()
plt.show()

# Step 3
# ----------------------------------------------------------------------------
# Calculate the Pearson correlation between the coordinates (X, Y, Z) and the target variable (Step)
correlation_matrix = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix between Coordinates and Maintenance Step')
plt.show() 

# Step 4
# ----------------------------------------------------------------------------
# Split the data into features (X) and target (y)
X = df[['X', 'Y', 'Z']]
y = df['Step']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
log_reg = LogisticRegression()
random_forest = RandomForestClassifier()
svc = SVC()

# Define hyperparameter grids for each model
param_grid_log_reg = {'C': [0.1, 1, 10]}
param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]}
param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# GridSearch for Logistic Regression
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)
grid_search_log_reg.fit(X_train, y_train)

# GridSearch for Random Forest
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# GridSearch for SVC
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5)
grid_search_svc.fit(X_train, y_train)

# RandomizedSearchCV example (for Random Forest)
random_search_rf = RandomizedSearchCV(random_forest, param_distributions=param_grid_rf, n_iter=5, cv=5, random_state=42)
random_search_rf.fit(X_train, y_train)

# Evaluate performance of the models on the test set
y_pred_log_reg = grid_search_log_reg.predict(X_test)
y_pred_rf = grid_search_rf.predict(X_test)
y_pred_svc = grid_search_svc.predict(X_test)

# Calculate accuracy
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svc = accuracy_score(y_test, y_pred_svc)

print(f'Logistic Regression Accuracy: {acc_log_reg}')
print(f'Random Forest Accuracy: {acc_rf}')
print(f'SVC Accuracy: {acc_svc}')

# Step 5: Model Performance Analysis
# ----------------------------------------------------------------------------
# Evaluate the Logistic Regression model
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_log_reg))

# Evaluate the Random Forest model
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

# Evaluate the SVC model
print("SVC Classification Report")
print(classification_report(y_test, y_pred_svc))

# Let's create a confusion matrix for the best-performing model (we'll assume Random Forest for this example)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 6: Stacked Model Performance Analysis
# ----------------------------------------------------------------------------
# Define the base models (Random Forest and SVC)
estimators = [
    ('rf', grid_search_rf.best_estimator_),  # Best Random Forest model from grid search
    ('svc', grid_search_svc.best_estimator_)  # Best SVC model from grid search
]

# Define the Stacking Classifier
stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the stacked model on the training set
stacked_model.fit(X_train, y_train)

# Make predictions with the stacked model on the test set
y_pred_stacked = stacked_model.predict(X_test)

# Evaluate the stacked model
print("Stacked Model Classification Report")
print(classification_report(y_test, y_pred_stacked))

# Generate the confusion matrix for the stacked model
conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stacked, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 7: Model Evaluation - Save the trained stacked model
# -----------------------------------------------------------------------------
model_filename = 'stacked_model.joblib'

# Save the model
joblib.dump(stacked_model, model_filename)

print(f"Model saved as {model_filename}")

# ---- Now let's demonstrate how to load the saved model and make predictions ----
# Load the saved model
loaded_model = joblib.load(model_filename)

# Example: Predict the maintenance steps for new coordinate data
new_coordinates = [[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]]

# Predict the maintenance steps for these new coordinates
predictions = loaded_model.predict(new_coordinates)

print("Predicted Maintenance Steps for the new coordinates:", predictions)







