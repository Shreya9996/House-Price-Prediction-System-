import pandas as pd 
import numpy as np

data = pd.read_csv(r"C:\Users\Admin\Documents\Scikit _learn\house_price_prediction_550_rows.csv")
# print(data.columns)
# print(data.info())
# print(data.isnull().sum())
# data.drop_duplicates()
# print(data.duplicated().sum())

from sklearn.model_selection import train_test_split
x = data.drop("Price",axis=1)
y = data["Price"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train = pd.DataFrame(data=x_train_scaled,columns=x_train.columns)
x_test = pd.DataFrame(data=x_test_scaled,columns=x_test.columns)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score,mean_absolute_error

print("Accuracy : ",r2_score(y_test,y_pred))
print("Mean :",mean_absolute_error(y_test,y_pred))
print(data.columns)



def predict_price(Area_sqft,Bedrooms,Bathrooms,Floors,Parking,Age_of_House,Distance_from_City_km):

    data_frame =pd.DataFrame( {
        "Area_sqft" : [Area_sqft],
        "Bedrooms": [Bedrooms],
        "Bathrooms" :[Bathrooms],
        "Floors" : [Floors],
        "Parking" : [Parking],
        "Age_of_House" : [Age_of_House],
        "Distance_from_City_km" : [Distance_from_City_km]
    })

    # features = np.array([[Area_sqft,Bedrooms,Bathrooms,Floors,Parking,Age_of_House,Distance_from_City_km]])
    scaled_data = scaler.transform(data_frame)
    features  = pd.DataFrame(data=scaled_data,columns=x_train.columns)
    result = model.predict(features)
    return result[0]



Area_sqft = 1461
Bedrooms = 5
Bathrooms = 2
Floors = 2
Parking =  0 
Age_of_House = 18
Distance_from_City_km = 25

result = predict_price(Area_sqft,Bedrooms,Bathrooms,Floors,Parking,Age_of_House,Distance_from_City_km)
print(result)

