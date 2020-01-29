# -*- coding: utf-8 -*-
"""

"""
# Import all required libraries
import pandas as pd 
import numpy as np
#import os

# Load data set
df = pd.read_csv('Binance_BTCUSDT_1h.csv',header=0)#, nrows=1000)
# Transform all columns
df=df.drop(['Symbol'], axis=1)
#print(df.shape)

ndf=pd.DataFrame()

# Conver OHLC values
ndf=pd.concat([ndf,pd.DataFrame({'dO':np.diff(np.log(df.Open+1))})],  axis=1)
ndf=pd.concat([ndf,pd.DataFrame({'dH':np.diff(np.log(df.High+1))})],  axis=1)
ndf=pd.concat([ndf,pd.DataFrame({'dL':np.diff(np.log(df.Low+1))})],  axis=1)
ndf=pd.concat([ndf,pd.DataFrame({'dC':np.diff(np.log(df.Close+1))})],  axis=1)

# Convert data to categorical Data Frame
M=np.empty([df.shape[0]-1, 5], dtype='<U4')
for i in range(1,df.shape[0]):
    # split date and time
     V=df.Date[i]
     V=V.split(' ')
     YMD=np.asarray(V[0].split('-'))
     H=np.asarray(V[1].split('-'))
     V=np.hstack((YMD,H))
     M[i-1,]=V
M=pd.DataFrame(M)
     
M.columns=['Y','M','D','h','F']

# Add M to the base Data Frame
ndf=pd.concat([ndf,M],  axis=1)

# Dummy coding
ndf = pd.get_dummies(data=ndf, columns=['Y','M','D','h','F'], drop_first=True)
print("The initial dummy data table:",ndf.shape)


# !!!!!!!!!!!!!!!! Set parameter value !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Add prehistory with set deep value
deep=24 # 24 hours of prehistory
ndf2=ndf.copy()
for i in range(deep):
    s_ndf=ndf[deep:].copy()
    for j in range(i+1):
        s_ndf=s_ndf.append(pd.Series(), ignore_index=True)
    s_ndf.columns=s_ndf.columns+"_"+str(i+1)  
    ndf2=pd.concat([ndf2,s_ndf], axis=1, ignore_index=False)

# Drop NA
ndf2=ndf2.dropna()

#print(ndf.shape)
#print(ndf.head())
print("The dummy data table with prehistory:",ndf2.shape)


# Add class label
Target=(ndf2.iloc[:,ndf2.shape[1]-ndf.shape[1]-0]>0).astype(int).astype(object)
Target=Target[1:]
Target=np.append(Target,np.NaN)
ndf2['Target']=Target

# The row for predictions
prV=ndf2.iloc[ndf2.shape[0]-1,0:ndf2.shape[1]-1]
prV=np.transpose(pd.DataFrame(prV))
# get the current date and hour
cdt=df.iloc[df.shape[0]-1,0]
prV=pd.DataFrame(prV)
# add date and time
prV['dt']=str(cdt)

prV.to_csv(r'predict_features.csv',index = None, header=True)

# Drop NA again for delete the las row
ndf2=ndf2.dropna()

print("The final data table:",ndf2.shape)

# Save to file
ndf2.to_csv(r'input.csv',index = None, header=True)
