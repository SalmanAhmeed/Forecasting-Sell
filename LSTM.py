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
from keras.optimizers import Adam


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




########   Preparing the first dataset for Trainning And  testing the model   #############
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



############   Creating and Training the LSTM model   ##################

# Preparing training and testing data
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Preparing testing data from the Second Dataset
second_X_test_lstm = second_X_test.reshape(second_X_test.shape[0], 1, second_X_test.shape[1])

# Creating a Sequential model
model = Sequential()
#Adding an LSTM layer with 4 units,
model.add(LSTM(4, batch_input_shape=(1, X_train_lstm.shape[1], X_test_lstm.shape[2])))
# Adding a Dense layer with 10 units and ReLU activation function
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
# Creating a list of callbacks including EarlyStopping and the ModelCheckpoint
checkpoint_filepath = os.getcwd()
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, 
                                            monitor='val_loss', mode='min', save_best_only=True)
callbacks = [EarlyStopping(patience=5), model_checkpoint_callback]

#Training the model
history = model.fit(X_train_lstm, y_train, epochs=200, batch_size=1, 
                    validation_data=(X_test_lstm, y_test), callbacks=callbacks)


#########   Predicting test data for the second dataset  #############
# Testing  and making predictions using the trained LSTM model
lstm_pred = model.predict(X_test_lstm, batch_size=1)
lstm_pred = lstm_pred.reshape(-1,1)
lstm_pred_test_set = np.concatenate([lstm_pred,X_test], axis=1)
lstm_pred_test_set = scaler.inverse_transform(lstm_pred_test_set)
re_list = []
for index in range(0, len(lstm_pred_test_set)):
    re_list.append(lstm_pred_test_set[index][0] + actual_sales[index])
lstm_pred_series = pd.Series(re_list, name='lstm_pred')
predict_df = predict_df.merge(lstm_pred_series, left_index=True, right_index=True)


# Evaluate performance metrics
lstm_rmse = np.sqrt(mean_squared_error(predict_df['lstm_pred'], month_sales['sales'][-12:]))
lstm_mae = mean_absolute_error(predict_df['lstm_pred'], month_sales['sales'][-12:])
lstm_r2 = r2_score(predict_df['lstm_pred'], month_sales['sales'][-12:])
print('First Dataset RMSE: ', lstm_rmse)
print('First Dataset MAE: ', lstm_mae)
print('First Dataset R2 Score: ', lstm_r2)

plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(month_sales['date'], month_sales['sales'],color="green")
plt.plot(predict_df['date'], predict_df['lstm_pred'],color="black")
plt.title("LSTM Predict")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()



#########   Predicting test data for the second dataset  #############

# Testing  and making predictions using the trained LSTM model
lstm_pred = model.predict(second_X_test_lstm, batch_size=1)
lstm_pred = lstm_pred.reshape(-1,1)
lstm_pred_test_set = np.concatenate([lstm_pred,second_X_test], axis=1)
lstm_pred_test_set = scaler.inverse_transform(lstm_pred_test_set)
re_list = []
for index in range(0, len(lstm_pred_test_set)):
    re_list.append(lstm_pred_test_set[index][0] + second_actual_sales[index])
lstm_pred_series = pd.Series(re_list, name='lstm_pred_2')
predict_df = predict_df.merge(lstm_pred_series, left_index=True, right_index=True)


# Evaluate performance metrics
lstm_rmse = np.sqrt(mean_squared_error(predict_df['lstm_pred_2'], second_month_sales['sales'][-12:]))
lstm_mae = mean_absolute_error(predict_df['lstm_pred_2'], second_month_sales['sales'][-12:])
lstm_r2 = r2_score(predict_df['lstm_pred_2'], second_month_sales['sales'][-12:])
print('')
print('Second Dataset RMSE: ', lstm_rmse)
print('Second Dataset MAE: ', lstm_mae)
print('Second Dataset R2 Score: ', lstm_r2)

