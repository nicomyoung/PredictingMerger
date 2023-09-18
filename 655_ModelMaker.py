import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your DataFrame
merged_df = pd.read_csv('finalData.csv')
new_columns = [
    "word", "transcription", "merge_status", "original_transcript", "merger_phones",
    "target_Type", "target_f1", "target_f2", "target_f3",
    "goal_Type", "goal_f1", "goal_f2", "goal_f3",
    "preceding_Type", "preceding_f1", "preceding_f2", "preceding_f3",
    "following_Type", "following_f1", "following_f2", "following_f3"
]

# Load your DataFrame
merged_df = pd.read_csv('finalData.csv')

# Rename the columns
merged_df.columns = new_columns


# Create a new DataFrame excluding rows with 'None'
final_df = merged_df.dropna()
removed_rows = len(merged_df) - len(final_df)
print(f"Removed rows: {removed_rows}")

# Drop unnecessary columns
columns_to_drop = ['word', 'original_transcript', 'merger_phones']
final_df = final_df.drop(columns=columns_to_drop, axis=1)

print("Columns dropped, preparing the data for the machine learning model...")

# Prepare the data for the machine learning model
X = final_df.drop(['merge_status'], axis=1)
y = final_df['merge_status']

print("Performing one hot encoding...")

# One hot encoding
one_hot_encoder = OneHotEncoder()
X = one_hot_encoder.fit_transform(X)

print("Splitting the data into train and test sets...")

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print("Training the RandomForestClassifier...")

# Train the RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

print("Performing cross validation...")

# Cross Validation
cross_val_score = cross_val_score(random_forest, X_train, y_train, cv=5)
print(f"Cross Validation Score: {cross_val_score.mean()}")

print("Performing hyperparameter tuning...")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(random_forest, param_grid, cv=5, verbose = 2)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

print("Re-training the model with best parameters...")

# Re-train the model with best parameters
random_forest = RandomForestClassifier(**grid_search.best_params_)
random_forest.fit(X_train, y_train)

print("Making predictions and evaluating the model...")

# Make predictions
y_pred = random_forest.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Analyzing feature importance and visualization...")

# Feature Importance and Visualization
importances = random_forest.feature_importances_
feature_names = one_hot_encoder.get_feature_names_out()
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Comparing predicted and true values
df_results = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
sns.scatterplot(data=df_results, x='True', y='Predicted', hue='Predicted', style='True', palette='Set2')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('True vs Predicted Values')
plt.show()
