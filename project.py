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

#Step 3: Data Exploration
print(h_data.shape)
print(h_data.info())
print(h_data.describe())

#Step 4: Data Cleaning
print(h_data.isnull().sum())
h_data.dropna(inplace=True)

#Step 5: Data Visualization
sns.pairplot(h_data)
plt.show()

sns.heatmap(h_data.corr(), annot=True)
plt.show()

X = h_data[['Area', 'Bedrooms', 'Bathrooms']]
y = h_data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Values:", y_pred)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

joblib.dump(model, "house_price_model.pkl")

new_house = pd.DataFrame(
    [[1200, 3, 2]],
    columns=['Area', 'Bedrooms', 'Bathrooms']
)

prediction = model.predict(new_house)
print(prediction)