    
import pandas as pd
from datetime import datetime
import firebase
from firebase_admin import auth, firestore
from statsmodels.tsa.ar_model import AR
from random import random
import matplotlib.pyplot as plt
def predict():
    db= firestore.client()
    sensor_ref = db.collection('sensors').get()
    mq135values=[]
    mq135labels=[]
    mq7values=[]
    mq2values=[]
    pm2_5values=[]
    for item in sensor_ref:
        dict_item = item.to_dict()
        time=datetime.fromtimestamp(dict_item['timestamp']).strftime("%a %I:%M %p ")
        mq135labels.append(dict_item['timestamp'])
        mq135values.append(dict_item['mq135'])
        mq7values.append(dict_item['mq7'])
        mq2values.append(dict_item['mq2']) 
    sensorData=pd.DataFrame({
        'ds':mq135labels,
        'y':mq135values
    })

    sensorData.set_index('ds',inplace=True)
    sensorData.index = sensorData.index.astype('datetime64[ns]')
    # fit model
    model = AR(sensorData['y'])
    model_fit = model.fit()
    # make prediction
    size=len(sensorData['y'])
    yhat = model_fit.predict(size,size+24)
    print(yhat.tolist())
    print(yhat)
if __name__ == "__main__":
    predict()
