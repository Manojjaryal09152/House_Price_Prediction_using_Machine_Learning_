#Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


#create dataset
data={
    'Area':[1200,1500,1800],
    'Bedrooms':[2,3,3],
    'Bathrooms':[1,2,2],
    'Price':[2500000,3500000,4200000]

}
df = pd.DataFrame(data)
df
df.to_csv('house_data.csv',index=False)
#Step 2: Load Dataset

h_data=pd.read_csv('House_data.csv')
h_data.head() 