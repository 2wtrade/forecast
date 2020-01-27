from tensorflow import keras

import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

# Load input data set
df = pd.read_csv('predict_features.csv',header=0)#, nrows=1000)
print(df.shape)
# get current date
cdt=df['dt']
# delete the last value
df=df.iloc[:,0:(df.shape[1]-1)]

# Prdictor matrix
X = df.values
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Predict based on Deep MLP
# Load models
model1 = keras.models.load_model('model_deep_mlp.h5')
model2 = keras.models.load_model('model_lstm.h5')
# Summarize models
model1.summary()
model2.summary()
# Get predictions
pr1 = model1.predict(X[0])
pr2 = model2.predict(X)
class_1='The price will be higher'
class_2='The price will be higher'

# predictions based on DNN
if pr1>0.5: 
    print("The price will be higher with probability (MLP):",np.round(float(pr1),2))
    pr1=np.round(float(pr1),2)
if pr1<=0.5: 
    print("The price will be lower with probability: MLP)",np.round(1-float(pr1),2))
    class_1='The price will be lower'
    pr1=np.round(1-float(pr1),2)

# predictions based on LSTM
if pr2>0.5: 
    print("The price will be higher with probability (LSTM):",np.round(float(pr2),2))
    pr2=np.round(float(pr2),2)
if pr2<=0.5: 
    print("The price will be lower with probability: LSTM)",np.round(1-float(pr2),2))
    class_2='The price will be lower'
    pr2=np.round(1-float(pr2),2)

# save predictions
res1=pd.DataFrame({'Current_moment':cdt,"prediction":class_1,"probability":pr1})
res2=pd.DataFrame({'Current_moment':cdt,"prediction":class_2,"probability":pr2})
all_res=pd.concat([res1,res2])
all_res.index=['DNN',"LSTM"]
#output
all_res.to_csv('output.csv')
    