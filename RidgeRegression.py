import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint	
from tensorflow.keras.models import Sequential


##############   Preparing the second dataset for testing the model   #################

# Read and prepare the second dataset 
second_sale_dataset = pd.read_csv('Second_Sales_DataSet.csv')
second_sale_dataset = second_sale_dataset.drop(['store','item'], axis=1)
second_sale_dataset['date'] = pd.to_datetime(second_sale_dataset['date'])
second_sale_dataset['date'] = second_sale_dataset['date'].dt.to_period('M')
second_month_sales = second_sale_dataset.groupby('date').sum().reset_index()
second_month_sales['date'] = second_month_sales['date'].dt.to_timestamp()
second_month_sales['sales_diff'] = second_month_sales['sales'].diff()
second_month_sales = second_month_sales.dropna()   

# finding sell deferences and preparing supverised data
second_supverised_sale_data = second_month_sales.drop(['date','sales'], axis=1)
for i in range(1,13):
    column_name = 'month_' + str(i)
    second_supverised_sale_data[column_name] = second_supverised_sale_data['sales_diff'].shift(i)



#Preparing supervised data
second_supverised_sale_data = second_supverised_sale_data.dropna().reset_index(drop=True)  #remove the null value 
second_train_sale_data = second_supverised_sale_data[:-22]
second_test_sale_data = second_supverised_sale_data[-12:]

# Fitting the scaler on the training data and transforming the data
second_scaler = MinMaxScaler(feature_range=(-1,1))
second_scaler.fit(second_test_sale_data)
second_test_sale_data = second_scaler.transform(second_test_sale_data)

# Extracting testing data 
second_X_test = second_test_sale_data[:,1:]

second_sales_dates = second_month_sales['date'][-12:].reset_index(drop=True)
second_actual_sales = second_month_sales['sales'][-13:].to_list()




########   Preparing the dataset for Trainning And  testing the model   #############
# Read and prepare the dataset 
sale_dataset = pd.read_csv('First_Sales_DataSet.csv')
sale_dataset = sale_dataset.drop(['store','item'], axis=1)
sale_dataset['date'] = pd.to_datetime(sale_dataset['date'])
sale_dataset['date'] = sale_dataset['date'].dt.to_period('M')
month_sales = sale_dataset.groupby('date').sum().reset_index()
month_sales['date'] = month_sales['date'].dt.to_timestamp()

# Finding sell deferences
month_sales['sales_diff'] = month_sales['sales'].diff()
month_sales = month_sales.dropna()   

# Preparing supverised data
supverised_sale_data = month_sales.drop(['date','sales'], axis=1)
for i in range(1,13):
    column_name = 'month_' + str(i)
    supverised_sale_data[column_name] = supverised_sale_data['sales_diff'].shift(i)
supverised_sale_data = supverised_sale_data.dropna().reset_index(drop=True)  #remove the null value 

# Extracting training and testing data 
train_sale_data = supverised_sale_data[:-12]
test_sale_data = supverised_sale_data[-12:]

# Fitting the scaler on the training data and transforming the data 
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_sale_data)
train_sale_data = scaler.transform(train_sale_data)
test_sale_data = scaler.transform(test_sale_data)

# Extracting features and labels for training data
X_train, y_train = train_sale_data[:,1:], train_sale_data[:,0:1]

# Extracting features and labels for testing data
X_test, y_test = test_sale_data[:,1:], test_sale_data[:,0:1]

# convert to 1-dimensional arrays
y_train = y_train.ravel()  
y_test = y_test.ravel() 

sales_dates = month_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
actual_sales = month_sales['sales'][-13:].to_list()




###############   Creating and Training the Ridge Regression  ##################

# Creating and Training the model 
ridge_model = Ridge(alpha=1.0)  
ridge_model.fit(X_train, y_train)


#########   Predicting test data for the first dataset  #############
# Making predictions using the trained model
ridge_pred = ridge_model.predict(X_test)

# Inverse transform to get predictions in original scale
ridge_pred = ridge_pred.reshape(-1, 1)
ridge_pred_test_set = np.concatenate([ridge_pred, X_test], axis=1)
ridge_pred_test_set = scaler.inverse_transform(ridge_pred_test_set)

# Combine predictions with actual sales
result_list_ridge = []       
for index in range(0, len(ridge_pred_test_set)):
    result_list_ridge.append(ridge_pred_test_set[index][0] + actual_sales[index])

# Merge with the prediction DataFrame
ridge_pred_series = pd.Series(result_list_ridge, name='ridge_pred')
predict_df = predict_df.merge(ridge_pred_series, left_index=True, right_index=True)

# Evaluate performance metrics
ridge_rmse = np.sqrt(mean_squared_error(predict_df['ridge_pred'], month_sales['sales'][-12:]))
ridge_mae = mean_absolute_error(predict_df['ridge_pred'], month_sales['sales'][-12:])
ridge_r2 = r2_score(predict_df['ridge_pred'], month_sales['sales'][-12:])
print('First Dataset RMSE: ', ridge_rmse)
print('First Dataset MAE: ', ridge_mae)
print('First Dataset R2 Score: ', ridge_r2)

# Visualizing predicted results
plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(month_sales['date'], month_sales['sales'],color="green")
plt.plot(predict_df['date'], predict_df['ridge_pred'],color="blue")
plt.title("Ridge Regression Predict")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()



#########   Predicting test data for the second dataset  #############
# Making predictions using the trained model
ridge_pred = ridge_model.predict(second_X_test)

# Inverse transform to get predictions in original scale
ridge_pred = ridge_pred.reshape(-1, 1)
ridge_pred_test_set = np.concatenate([ridge_pred, second_X_test], axis=1)
ridge_pred_test_set = scaler.inverse_transform(ridge_pred_test_set)

# Combine predictions with actual sales
result_list_ridge = []       
for index in range(0, len(ridge_pred_test_set)):
    result_list_ridge.append(ridge_pred_test_set[index][0] + second_actual_sales[index])

# Merge with the prediction DataFrame
ridge_pred_series = pd.Series(result_list_ridge, name='ridge_pred_2')
predict_df = predict_df.merge(ridge_pred_series, left_index=True, right_index=True)

# Evaluate performance metrics
ridge_rmse = np.sqrt(mean_squared_error(predict_df['ridge_pred_2'], second_month_sales['sales'][-12:]))
ridge_mae = mean_absolute_error(predict_df['ridge_pred_2'], second_month_sales['sales'][-12:])
ridge_r2 = r2_score(predict_df['ridge_pred_2'], second_month_sales['sales'][-12:])

print('')
print('Second Dataset RMSE: ', ridge_rmse)
print('Second Dataset MAE: ', ridge_mae)
print('Second Dataset R2 Score: ', ridge_r2)









