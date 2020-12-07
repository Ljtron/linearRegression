import requests
import pandas as pd 
import io
from linearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# The dataset that is going to train the ai
df = pd.read_csv("https://raw.githubusercontent.com/Baakchsu/LinearRegression/master/weight-height.csv")
#print(df.head())
reg = LinearRegression()

def preprocess():
    x = (df['Weight']-df['Weight'].mean())/df['Weight'].std() #standardization of the dataset
    y = (df['Height']-df['Height'].mean())/df['Height'].std() #standardization of the dataset
    return(x, y)

def train():
    (x, y) = preprocess()
    (weights,intercept,parameter_cache) = reg.train(x[:-180],y[:-180], 500)
    print(parameter_cache)

def prediction():
    (x, y) = preprocess()
    pred = reg.predict(np.array(x[-180:]))
    plt.scatter(x[-180:],y[-180:])
    plt.plot(x[-180:],pred)
    plt.show()

train()
prediction()
