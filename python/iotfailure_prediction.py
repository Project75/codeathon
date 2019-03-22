
# coding: utf-8

# # IoT Equipment Failure Prediction using Sensor data
# ##  Environment Setup
# ###  Import dependent libraries

# In[26]:


# Import libraries
import pandas as pd
import numpy as np
import pdb
import json
import re
import requests
import sys
import types
#import ibm_boto3


# In[27]:


# Import libraries
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from botocore.client import Config


# ##  Create IoT Predictive Analytics Functions

# In[28]:


# Function to extract Column names of dataset
def dataset_columns(dataset):
    return list(dataset.columns.values)

# Function to train Logistic regression model
def train_logistic_regression(x_vals, y_vals):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(x_vals, y_vals)
    return logistic_regression_model

# Function to return Predicted values
def score_data(trained_model, x_vals):
    ypredict = trained_model.predict(x_vals)
    return ypredict

# Function to calculate Prediction accuracy of model
def model_accuracy(trained_model, variables, targets):
    accuracy_score = trained_model.score(variables, targets)
    return accuracy_score

# Function to generate Confusion matrix
def confusion_matrix(actfail, predictfail):
  # Compute Confusion matrix
  print("Actual, Predicted Observations:  ",len(actfail), len(predictfail))
  # print(actfail, predictfail)
  anpn = 0
  anpy = 0
  aypn = 0
  aypy = 0
  
  for i in range(len(actfail)):
      if (actfail[i]==0 and predictfail[i]==0):
          anpn = anpn + 1
      elif (actfail[i]==0 and predictfail[i]==1):
          anpy = anpy + 1
      elif (actfail[i]==1 and predictfail[i]==0):
          aypn = aypn + 1
      else:
          aypy = aypy + 1
  # Confusoin matrix
  print ("--------------------------------------------")
  print ("Confusion Matrix")
  print ("--------------------------------------------")
  print ("              ", "Predicted N", "Predicted Y")
  print ("Actual N      ", anpn,"          ", anpy) 
  print ("Actual Y      ", aypn,"          ", aypy)
  print ("--------------------------------------------")
  print ("Total observations  :  ", anpn+anpy+aypn+aypy)
  print ("False Positives     :  ", anpy)
  print ("False Negatives     :  ", aypn)
  print ("Overall Accuracy    :  ", round((float(anpn+aypy)/float(anpn+anpy+aypn+aypy))*100, 2), "%")
  print ("Sensitivity/Recall  :  ", round((float(aypy)/float(aypn+aypy))*100, 2), "%")
  print ("Specificity         :  ", round((float(anpn)/float(anpn+anpy))*100, 2), "%")
  print ("Precision           :  ", round((float(aypy)/float(anpy+aypy))*100, 2), "%")
  print ("--------------------------------------------")


# In[29]:



df_iotdata = pd.read_csv('../data/iot_sensor_dataset.csv')

# Check Number of observations read for analysis
print ("Number of Observations :", len(df_iotdata))
# Inspect a few observations
df_iotdata.head()


# In[30]:


# Print dataset column names
datacolumns = dataset_columns(df_iotdata)
print ("Data set columns : ", list(datacolumns))


# In[31]:


### Feature extraction


# In[32]:


v_feature_list = ['atemp', 'PID', 'outpressure', 'inpressure', 'temp']
v_target = 'fail'
v_train_datasize = 0.7


# ### Train test split

# In[33]:


# Split Training and Testing data
train_x, test_x, train_y, test_y = train_test_split(df_iotdata[v_feature_list], df_iotdata[v_target], train_size=0.7)
print ("Train x counts : ", len(train_x), len(train_x.columns.values))
print ("Train y counts : ", len(train_y))
 
print ("Test x counts : ", len(test_x), len(test_x.columns.values))
print ("Test y counts : ", len(test_y))


# ###  Train the Predictive model

# In[34]:


# Training Logistic regression model
trained_logistic_regression_model = train_logistic_regression(train_x, train_y)

train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)

# Testing the logistic regression model
test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)

print ("Training Accuracy : ", round(train_accuracy * 100, 2), "%")


# ### Score the Test data using the Trained model

# In[35]:


# Model accuracy: Score and construct Confusion matrix for Test data
actfail = test_y.values
predictfail = score_data(trained_logistic_regression_model, test_x)


# In[ ]:


predictfail


# ##  Confusion matrix for deeper analysis of Prediction accuracy
# #####   Confusion matrix outputs below can be used for calculating more customised Accuracy metrics

# In[36]:


# Print Count of Actual fails, Predicted fails
# Print Confusion matrix
confusion_matrix(actfail, predictfail)


# In[38]:


get_ipython().system('jupyter nbconvert --to script iotfailure_prediction.ipynb')

