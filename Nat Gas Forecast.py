import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet


natgas=pd.read_csv("C:/Users/alich/Desktop/Python/Finance/Forage JPMorgan Job Simulations/Quantitative Research/Nat_Gas.csv"
                   , parse_dates=["Dates"])
natgas.index=natgas["Dates"] #Define the index for the natgas dataframe to be the date column
natgas=natgas.drop(columns=["Dates"]) #drop the date column, the dates are encoded in the index
newindex=pd.date_range(natgas.index[0],natgas.index[-1]) #define a new range of dates to be every day 
                                                         #from the first date to the last date provided
natgas=natgas.reindex(newindex, fill_value=np.nan)#Set the new index for natgas to be the range just created,
                                                  #and fill all of the missing price entries with NaN
                                                 
natgas=natgas.interpolate("cubic")#Replace the NaN entries with interpolated prices between the provided entries
trend=natgas.rolling(356).mean()#There's a roughly yearly seasonality to the data; calculate the trend according to this

decomp=sm.tsa.seasonal_decompose(natgas, model="additive")
plot_acf(decomp.resid) #0 autocorrelation between the residuals after decomposing additively
plt.title("Residuals Autocorrelation - Additive")
plt.grid()
plt.show()

decomp=sm.tsa.seasonal_decompose(natgas, model="multiplicative")
plot_acf(decomp.resid) #0 autocorrelation between the residuals after decomposing multiplicatively
plt.title("Residuals Autocorrelation - Multiplicative")
plt.grid()
plt.show()
#Since there's no autocorrelation regardless of whether we decompose additively or multiplicatively,
#it doesn't matter which we choose. We'll choose to run Prophet in seasonality_mode = "multiplicative".

plt.figure(figsize=(10,6))#Plot the price data
plt.plot(natgas)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Natural Gas Price History")
plt.grid()
plt.show()

plt.figure(figsize=(10,6))#Plot the trend
plt.plot(trend)
plt.title("Trend in Natural Gas Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid()
plt.show()

#At this point, we have an estimated price for any point in the range of the provided data. We have provided more
#granularity to the client. However, in case the client wants an estimate of future prices, we are now going to
#perform some time series forecasting to forecast the prices 1 year into the future, past the final date provided.


forecaster=Prophet(changepoint_prior_scale=0.0025, seasonality_mode="multiplicative")#Initialise the Prophet model for time series forecasting
horizon=365#Forecast 365 days into the future

y_train=natgas[:-horizon] #Let the training data be everything from the beginning of natgas until 365 days from the end
y_test=natgas.tail(horizon)#Let the testing data be the last 365 days of prices.

forecaster.fit(y_train)#Fit our initialised model to the training data
fh=ForecastingHorizon(y_test.index, is_relative=False)#Define our forecasting horizon to be the days in the testing index;
                                                      #namely we predict the last 365 days of data. This is to compare the
                                                      #predictions to the known results.

#y_pred=forecaster.predict(fh)#Predict on our forecasting horizon
#ci=forecaster.predict_interval(fh,coverage=0.5)#Give a confidence interval for the prediction

#y_true=natgas.tail(horizon)#The true values are the last 365 days of the natgas dataframe

#plt.figure(figsize=(10,6))
#plt.plot(natgas.tail(horizon*3), label="Actual", color="black")
#plt.gca().fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color="b", alpha=0.1)
#plt.plot(y_pred, label="Predicted")
#plt.title("Natural Gas Prices - Test")
#plt.ylim(bottom=9.5)
#plt.grid()
#plt.legend()
#plt.show()


model=Prophet(changepoint_prior_scale=0.0025, seasonality_mode="multiplicative")
model.fit(natgas)
last_date=natgas.index.max()
future=ForecastingHorizon(pd.date_range(str(last_date), periods=horizon, freq="D"), is_relative=False)

y_pred=model.predict(future)
ci=model.predict_interval(future, coverage=0.5).astype("float")

plt.figure(figsize=(10,6))
plt.plot(natgas, label="Actual", color="black")
plt.gca().fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color="b", alpha=0.09)
plt.plot(y_pred, label="Predicted")
plt.title("Natural Gas Prices - Forecast (with confidence intervals)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.ylim(bottom=9.5)
plt.grid()
plt.legend()
plt.show()

forecasthistory=pd.concat([natgas, y_pred[1:]], axis=0) #first entry of y_pred is the prediction for 2024-09-30, which is an 
                                                        #already-existing data point. Therefore, concatenate from the 2nd entry.
forecasthistory.loc["2024-10-01":"2024-10-20"]=np.nan #Take out some days and interpolate to smooth the sudden jump
                                                      #between the end of the collected data and the start of the estimated data.
forecasthistory=forecasthistory.interpolate("cubic")
forecasthistory.to_csv("Forecast.csv", index=True)

plt.figure(figsize=(10,6))#Plot the forecasted data with the transition between available & forecasted data smoothed
plt.plot(forecasthistory[:-horizon], color = "black", label="History")
plt.plot(forecasthistory[-horizon:], color = "royalblue", label="Forecast")
plt.title("Natural Gas Prices - Forecast (Smoothed)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.ylim(bottom=9.5)
plt.grid()
plt.legend()
plt.show()

date_input=input("Enter Date (YYYY-MM-DD):")
print("Estimated Price:", float(forecasthistory.loc[f"{date_input}"]))



