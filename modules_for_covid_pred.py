# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:09:49 2022

@author: End User
"""
import matplotlib.pyplot as plt
import numpy as np


class EDA():
    def __init__(self):
        pass
    def plot_graph(self,con_columns,df):
        for con in con_columns:
            plt.figure()
            plt.plot(df[con])
            plt.plot(df['cases_new'])
            plt.legend([con,'cases_new'])
            plt.title(con)
            plt.show()
    def append_list(self,win_size,df_new,X_train,y_train):
        for i in range(win_size,np.shape(df_new)[0]):
            X_train.append(df_new[i-win_size:i,0])
            y_train.append(df_new[i,0])

class model_evaluation():
    def __init__(self):
        pass
    def plot_hist_keys(self,hist):
        plt.figure()
        plt.plot(hist.history['mape'])
        plt.title('mape')
        plt.show()

        plt.figure()
        plt.plot(hist.history['loss'])
        plt.title('loss')
        plt.show()

class model_deployment():
    def __init__(self):
        pass
    def plotting_graph(self,test_df,predicted,mms):
        plt.figure()
        plt.plot(test_df,'b',label='actual_covid_cases')
        plt.plot(predicted,'r--',label='predicted_covid_cases')
        plt.title('Scaled')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='actual_covid_cases')
        plt.plot(mms.inverse_transform(predicted),'r--',label='predicted_covid_cases')
        plt.title('Inverse')
        plt.legend()
        plt.show()



















