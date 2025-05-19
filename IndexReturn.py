import pandas as pd
import matplotlib.pyplot as plt

#!pip install openpyxl

# Load the CSV data
data1 = pd.read_csv('new_S&P500-raw_prices.csv')
data2 = pd.read_csv('new_Russell 2000-raw_prices.csv')
data3 = pd.read_csv('new_Nasdaq-raw_prices.csv')
data4 = pd.read_csv('new_Dow-Jones-raw_prices.csv')
data5 = pd.read_csv('US 10 year Treasury-raw_prices.csv')

data1 = data1[['Date', 'Adj Close']]
data2 = data2[['Date', 'Adj Close']]
data3 = data3[['Date', 'Adj Close']]
data4 = data4[['Date', 'Adj Close']]
data5 = data5[['Date', 'Adj Close']]

data1['Date'] = pd.to_datetime(data1['Date'])
data2['Date'] = pd.to_datetime(data2['Date'])
data3['Date'] = pd.to_datetime(data3['Date'])
data4['Date'] = pd.to_datetime(data4['Date'])
data5['Date'] = pd.to_datetime(data5['Date'])

data1.set_index('Date', inplace=True)
data2.set_index('Date', inplace=True)
data3.set_index('Date', inplace=True)
data4.set_index('Date', inplace=True)
data5.set_index('Date', inplace=True)

full_date_range = pd.date_range(start=data1.index.min(), end=data1.index.max(), freq='D')

data1 = data1.reindex(full_date_range)
data2 = data2.reindex(full_date_range)
data3 = data3.reindex(full_date_range)
data4 = data4.reindex(full_date_range)
data5 = data5.reindex(full_date_range)

data1.ffill(inplace=True)
data2.ffill(inplace=True)
data3.ffill(inplace=True)
data4.ffill(inplace=True)
data5.ffill(inplace=True)

data1.reset_index(inplace=True)
data2.reset_index(inplace=True)
data3.reset_index(inplace=True)
data4.reset_index(inplace=True)
data5.reset_index(inplace=True)

data1.rename(columns={'index': 'Date'}, inplace=True)
data2.rename(columns={'index': 'Date'}, inplace=True)
data3.rename(columns={'index': 'Date'}, inplace=True)
data4.rename(columns={'index': 'Date'}, inplace=True)
data5.rename(columns={'index': 'Date'}, inplace=True)

data1.rename(columns={'Adj Close': 'S&P500'}, inplace=True)
data2.rename(columns={'Adj Close': 'Russell 2000'}, inplace=True)
data3.rename(columns={'Adj Close': 'Nasdaq'}, inplace=True)
data4.rename(columns={'Adj Close': 'Dow Jones'}, inplace=True)
data5.rename(columns={'Adj Close': 'US 10 Year Treasury'}, inplace=True)

df = pd.merge(data1, data2, on = "Date")
df = pd.merge(df, data3, on = "Date")
df = pd.merge(df, data4, on = "Date")
df = pd.merge(df, data5, on = "Date")

df.to_excel('IndexReturn.xlsx', engine='openpyxl', index=False)