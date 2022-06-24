# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:08:45 2022

@author: End User
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.utils import plot_model
from modules_for_covid_pred import EDA,model_evaluation,model_deployment


#%% Static
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')
CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')
CASES_NEW_MMS_PATH = os.path.join(os.getcwd(),'saved_models','cases_new_mms.pkl')
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')

#%% EDA
    # Step 1) Data loading
df = pd.read_csv(CSV_PATH)

#%% Backup data
# To copy original dataset into new dataframe (df_copy)
df_copy = df.copy()

#%%
    # Step 2) Data inspection
# To observe the dataset generally
df.info()
temp = df.describe().T
# Date column might be considered to be removed as it is unnecessary
# Presence of many missing values in dataset.
# Target column (cases_new) has missing values

# To check missing values in dataset
df.isna().sum()
df.isnull().sum()
# There are several rows in target column that cannot be detected as
# missing values. Therefore, we will making a list of missing value types.
missing_values = ["?"," "]
df = pd.read_csv(CSV_PATH,na_values=missing_values)
# Check again the presence of missing values
df.isnull().sum()
# Now, instead of 7 columns have missing values, there are 8 columns inspected
# to have missing values. The columns taht have NaNs are:
    # 1) cases_new --> 12
    # 2) all cluster columns --> 342
# Check for duplicates in dataset
df.duplicated().sum()
# There are no duplicates in the dataset

# Plot the graph of each columns. Every columns in the dataset set 
# are continuous
column_names = df.columns
con_columns = column_names[(df.dtypes=='int64')|(df.dtypes=='float64')]

eda = EDA()
eda.plot_graph(con_columns, df)

# cases_adult imitate the graph of cases_new. The early hypothesis is, this 
# we might say cases_adult contribute the most for the arising of cases_new.
# The number of unvaccinated (cases_invax) is the highest at late 500 days
# then, followed by partial vaccinated at late 500 days and lastly, the
# number of fully vaccinated going up and a little bit higher than partial 
# vaccinated later at early 600 days.
# This might be due to increase number of vaccination provided by government
# and public alertness against the danger of COVID.
# As we can see, the emerging of new cases arise steeply when unvaccinated 
# cases at the highest peak and then it takes an effect by going down 
# after the number of fully vaccinated increases.
# Nevertheless, the number of recovered were on par with the emerging of new
# cases.

    # Step 3) Data cleaning
    # Things to be filtered:
        # 1) Remove date column (unnecessary)
        # 2) Handle missing values (interpolate - cases_new)
    # By the way, since we only want to predict cases_new only, therefore, we
    # deal with cases_new only, hence, we will handle data cleaning just for 
    # cases_new. Otherwise, the rest will be unuseful. So, for cases_new has 
    # missing values, we will interpolate tha data to handle the missing values 
    # for cases_new. Then, the rest, we will left them as they are.

# Interpolate cases_new column
df_new = df['cases_new'].interpolate()

    # Step 4) Features selection
        # We are now selecting only cases_new data

    # Step 5) Data preprocessing
        # We will be using MinMaxScaler to ensure the data within the range 
        # 0-1. The reasin is because we want to preserve the shape of the 
        # original distribution.
# Scalling the data
mms = MinMaxScaler()
df_new = mms.fit_transform(np.expand_dims(df_new,axis=-1))

# Create empty list
X_train = []
y_train = []

# Set window size to 30
win_size = 30

# To append the list of X_train and y_train
eda.append_list(win_size, df_new, X_train, y_train)

# for i in range(win_size,np.shape(df_new)[0]):
#     X_train.append(df_new[i-win_size:i,0])
#     y_train.append(df_new[i,0])

# Change the list type into array
X_train = np.array(X_train)
y_train = np.array(y_train)

# To increase dimension --> 3 dimensions
X_train = np.expand_dims(X_train,axis=-1)   # Activate only once
#%% Model Development
model = Sequential()
model.add(Input(shape=(np.shape(X_train)[1],1))) # input_length # features
model.add(LSTM(32,return_sequences=(True)))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu')) # Output layer
model.summary()

model.compile(optimizer='adam',loss='mse',metrics='mape')
# plot_model(model,show_layer_names=(True),show_shapes=(True))

# callbacks - TensorBoard
tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)

# Model Training
hist = model.fit(X_train,y_train,batch_size=16,epochs=100,callbacks=[tensorboard_callback])

#%% Model Evaluation
# To plot the graph
hist.history.keys()
mod_evaluate = model_evaluation()
mod_evaluate.plot_hist_keys(hist)

#%% Model Deployment and Analysis
test_df = pd.read_csv(CSV_TEST_PATH)

# Check duplicates and missing values
test_df.duplicated().sum()
test_df.isnull().sum()
# cases_new has 1 missing value --> interpolate the dataset to handle the NaN

# Interpolate the target column
test_df['cases_new'] = test_df['cases_new'].interpolate()

# Check again the presence of NaN
test_df.isnull().sum()
# There is no more NaN presence in the dataset.

#%%
# Adjust the dimension of test_df to be matched with the dimension of df_new
test_df = test_df['cases_new']
test_df = mms.transform(np.expand_dims(test_df,axis=-1))

# concatenate test_df with df_new
concat_test = np.concatenate((df_new,test_df),axis=0)
concat_test = concat_test[-(win_size+len(test_df)):]

X_test = []
for i in range(win_size,len(concat_test)):
    X_test.append(concat_test[i-win_size:i,0])

# convert the type of X_test from list to array
X_test = np.array(X_test)

# Prediction
predicted = model.predict(np.expand_dims(X_test,axis=-1))

#%% Plotting graph
# To plot graph of actual cases against predicted cases for scaled values and
# actual values.
mod_deploy = model_deployment()
mod_deploy.plotting_graph(test_df, predicted, mms)

# Use inverse MinMaxScaler when plotting the graph to show the actual
# numbers/values

#%% Test Model Performance - MSE, MAE, MAPE
y_true = test_df
y_pred = predicted

print('MAE: '+str(mean_absolute_error(y_true,y_pred)))
print('MSE: '+str(mean_squared_error(y_true,y_pred)))
print('MAPE: '+str(mean_absolute_percentage_error(y_true,y_pred)))

y_true_inversed = mms.inverse_transform(y_true)
y_pred_inversed = mms.inverse_transform(y_pred)

print('MAE_inversed: '+str(mean_absolute_error(y_true_inversed,y_pred_inversed)))
print('MSE_inversed: '+str(mean_squared_error(y_true_inversed,y_pred_inversed)))
print('MAPE_inversed: '+str(mean_absolute_percentage_error(y_true_inversed,y_pred_inversed)))

#%% Discussion

# The model is able to predict the trend of the covid new cases.
# Meanwhile the mean absolute percentage error successfully achieved 0.16%
# which is below 1% when tested against testing dataset.
# Therefore, there is still a room for improvement. For example, we may
# implement the usage of bidirectional LTSM, use clustering to improve the 
# the accuracy of our model and other appropriate methods.


#%% Saving Models
# Saving MinMaxScaler as pickle
with open(CASES_NEW_MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

# Saving Model to .h5 file
model.save(MODEL_SAVE_PATH)

#%% Test (Manual calculation of MAPE)
print((mean_absolute_error(y_true,y_pred)/sum(abs(y_true)))*100)





