import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

######################################################################################Create Dataset

# Define the questions and their weights
questions = [
    "Are you exhausted in the morning at the thought of another day at work?",
    "How often do you feel worn out?",
    "Do you feel that every working hour is tiring for you?",
    "How often do you think: 'I can’t take it anymore'?",
    "How often are you emotionally exhausted?",
    "Does your work frustrate you?",
    "How often do you feel tired?",
    "Do you feel worn out at the end of the working day?",
    "Do you have enough energy for family and friends during your leisure time?",
    "How often are you physically exhausted?",
    "How often do you feel weak and susceptible to illness?",
    "Is your work emotionally exhausting?",
    "Do you feel burnt out because of your work?"
]

weights = [3, 2, 3, 4, 5, 2, 3, 4, 2, 4, 3, 5, 5]

combine_output = 0

for i in range (0, len(weights)):
  weights[i] = weights[i]*10


# Generate random responses for each participant
np.random.seed(42)  # For reproducibility
responses = np.random.randint(1, 6, size=(10000, len(questions)))

# Calculate total score for each participant
# Total Score = (Q1 * Weight1) + (Q2 * Weight2) + ... + (Qn * Weightn)
total_scores = np.dot(responses, weights)

# Calculate percentage likely to burnout
max_score = sum([5 * w for w in weights])
percentages = (total_scores / max_score) * 100

# Create a DataFrame
data = pd.DataFrame(responses, columns=questions)
data['Total Score'] = total_scores
data['Percentage Likely to Burnout'] = percentages

# Export to CSV
data.to_csv('copenhagen_burnout_inventory_dataset.csv', index=False)



################################################################################## Random Forest



# Load the dataset
data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Define features (X) and target variable (y)
X = data.drop(['Total Score', 'Percentage Likely to Burnout'], axis=1)
y = data['Percentage Likely to Burnout']

if combine_output == True:
  X = np.column_stack((X, total_scores))     #Uncomment to combime Actual output with input set to increate accuracy 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Save the model
joblib.dump(rf_regressor, 'rf_burnout_model.joblib')


############################################################################################### Linear Regrassion



# Load the dataset
data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Define features (X) and target variable (y)
X = data.drop(['Total Score', 'Percentage Likely to Burnout'], axis=1)
y = data['Percentage Likely to Burnout']

if combine_output == True:
  X = np.column_stack((X, total_scores))     #Uncomment to combime Actual output with input set to increate accuracy 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error:", mse)
print("R-squared:", r2)

joblib.dump(model, 'LR_burnout_model.joblib')


############################################################################################### Vetor Regression


# Load the dataset
data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Define features (X) and target variable (y)
X = data.drop(['Total Score', 'Percentage Likely to Burnout'], axis=1)
y = data['Percentage Likely to Burnout']

if combine_output == True:
  X = np.column_stack((X, total_scores))     #Uncomment to combime Actual output with input set to increate accuracy 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVR(kernel='linear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error:", mse)
print("R-squared:", r2)

joblib.dump(model, 'VR_burnout_model.joblib')


###############################################################################################  Gradient boosting regrassion



# Load the dataset
data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Define features (X) and target variable (y)
X = data.drop(['Total Score', 'Percentage Likely to Burnout'], axis=1)
y = data['Percentage Likely to Burnout']

if combine_output == True:
  X = np.column_stack((X, total_scores))     #Uncomment to combime Actual output with input set to increate accuracy 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error:", mse)
print("R-squared:", r2)

joblib.dump(model, 'GBR_burnout_model.joblib')


############################################################################################### MLP


# Load the dataset
data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Define features (X) and target variable (y)
X = data.drop(['Total Score', 'Percentage Likely to Burnout'], axis=1)
y = data['Percentage Likely to Burnout']

if combine_output == True:
  X = np.column_stack((X, total_scores))     #Uncomment to combime Actual output with input set to increate accuracy 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error:", mse)
print("R-squared:", r2)

joblib.dump(model, 'MLP_burnout_model.joblib')


model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)




############################################################################################### Prediction

# Load the trained model
loaded_model_RF = joblib.load('rf_burnout_model.joblib')
loaded_model_LR = joblib.load('LR_burnout_model.joblib')
loaded_model_VR = joblib.load('VR_burnout_model.joblib')
loaded_model_GBR = joblib.load('GBR_burnout_model.joblib')
loaded_model_MLP = joblib.load('MLP_burnout_model.joblib')

loaded_model = loaded_model_MLP
# Example input row for prediction
idx = 570

if combine_output != True:
  input_row = X_test.iloc[idx]
else:
  input_row = X_test[idx]

#input_row = X_test.iloc[idx]
input_row = X_test.iloc[idx]

actual_percentage = data.loc[idx, 'Percentage Likely to Burnout']
print("Actu %:", actual_percentage)

# Predict burnout percentage RF
predicted_burnout_percentage = loaded_model_RF.predict([input_row])
print("RF %:", predicted_burnout_percentage)

# Predict burnout percentage LR
predicted_burnout_percentage = loaded_model_LR.predict([input_row])
print("LR %:", predicted_burnout_percentage)

# Predict burnout percentage VR
predicted_burnout_percentage = loaded_model_VR.predict([input_row])
print("VR %:", predicted_burnout_percentage)

# Predict burnout percentage GBR
predicted_burnout_percentage = loaded_model_GBR.predict([input_row])
print("GBR %:", predicted_burnout_percentage)

# Predict burnout percentage MLR
predicted_burnout_percentage = loaded_model_MLP.predict([input_row])
print("MLP %:", predicted_burnout_percentage)


############################################################################################### Plot Single data Prediction vs Actual from Test Set



# Load the dataset
data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Select a specific input row (for example, the first row)
input_row = X.iloc[0].values.reshape(1, -1)

if combine_output != True:
  input_row = X.iloc[0].values.reshape(1, -1)
else:
  input_row = X[0].reshape(1, -1)

# Predict the burnout percentage for the input row
predicted_percentage = loaded_model.predict(input_row)

# Get the actual burnout percentage for the input row
actual_percentage = data.loc[0, 'Percentage Likely to Burnout']

# Plot actual vs. predicted burnout percentage
plt.figure(figsize=(8, 6))
plt.scatter(actual_percentage, predicted_percentage, color='blue', label='Actual vs. Predicted')
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Burnout Percentage')
plt.ylabel('Predicted Burnout Percentage')
plt.title('Actual vs. Predicted Burnout Percentage')
plt.legend()
plt.grid(True)
plt.show()


###############################################################################################  Custom Test(Single Test Case)


mdl = joblib.load('VR_burnout_model.joblib')

input_row_x = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
print(input_row_x)
total_scores = np.dot(input_row_x, weights)

max_score = sum([5 * w for w in weights])
percentages = (total_scores / max_score) * 100
if combine_output == True:
  input_row_x = np.hstack((input_row_x, [[actual_percentage]]))

predicted_percentage_x = mdl.predict(input_row_x)
print(predicted_percentage_x)



input_row_y = [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]
print(input_row_y)
total_scores = np.dot(input_row_y, weights)

max_score = sum([5 * w for w in weights])
percentages = (total_scores / max_score) * 100
if combine_output == True:
  input_row_y = np.hstack((input_row_y, [[actual_percentage]]))

predicted_percentage_y = mdl.predict(input_row_y)
print(predicted_percentage_y)

###############################################################################################  


data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Define features (X) and target variable (y)
X = data.drop(['Total Score', 'Percentage Likely to Burnout'], axis=1)
y = data['Percentage Likely to Burnout']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42),
    "SVR": SVR(kernel='linear')
}

# Dictionary to store predicted values
predictions = {}

# Train and predict for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

# Plot actual vs predicted for each model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='black', label='Actual', alpha=0.5)  # Plotting the line y = x
for name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, label=name)
plt.xlabel('Actual Percentage Burnout')
plt.ylabel('Predicted Percentage Burnout')
plt.title('Actual vs Predicted Percentage Burnout')
plt.legend()
plt.grid(True)
plt.show()

###############################################################################################  Comparison between Different models



data = pd.read_csv('copenhagen_burnout_inventory_dataset.csv')

# Define features (X) and target variable (y)
X = data.drop(['Total Score', 'Percentage Likely to Burnout'], axis=1)
y = data['Percentage Likely to Burnout']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42),
    "SVR": SVR(kernel='linear')
}

# Dictionary to store predicted values
predictions = {}

# Train and predict for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

# Plot actual vs predicted for each model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='black', label='Actual', alpha=0.5)  # Plotting the line y = x
for name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, label=name)
plt.xlabel('Actual Percentage Burnout')
plt.ylabel('Predicted Percentage Burnout')
plt.title('Actual vs Predicted Percentage Burnout')
plt.legend()
plt.grid(True)
plt.show()

###############################################################################################   Create and Save Test Report in csv

# Train and predict for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

# Combine actual and predicted values into a DataFrame
df = pd.DataFrame({'Actual': y_test})
for name, y_pred in predictions.items():
    df[name] = y_pred

# Save DataFrame to CSV
df.to_csv('model_predictions.csv', index=False)

print("Model predictions saved to 'model_predictions.csv'")

###############################################################################################

