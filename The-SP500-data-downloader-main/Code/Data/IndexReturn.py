import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
data1 = pd.read_csv('new_S&P500-raw_prices.csv')
data2 = pd.read_csv('new_Nasdaq-raw_prices.csv')
data3 = pd.read_csv('new_Russell 2000-raw_prices.csv')
data4 = pd.read_csv('new_Dow-Jones-raw_prices.csv')
data5 = pd.read_csv('US 10 year Treasury-raw_prices.csv')

data1 = data1[['Price', 'Close']]
data2 = data2[['Price', 'Close']]
data3 = data3[['Price', 'Close']]
data4 = data4[['Price', 'Close']]
data5 = data5[['Price', 'Close']]

data1 = data1.iloc[2:]
data2 = data2.iloc[2:]
data3 = data3.iloc[2:]
data4 = data4.iloc[2:]
data5 = data5.iloc[2:]

# data1 = data1.rename(index={'Price': 'Date'})
# data2 = data2.rename(index={'Price': 'Date'})
# data3 = data3.rename(index={'Price': 'Date'})
# data4 = data4.rename(index={'Price': 'Date'})
# data5 = data5.rename(index={'Price': 'Date'})

data1['Price'] = pd.to_datetime(data1['Price'])
data2['Price'] = pd.to_datetime(data2['Price'])
data3['Price'] = pd.to_datetime(data3['Price'])
data4['Price'] = pd.to_datetime(data4['Price'])
data5['Price'] = pd.to_datetime(data5['Price'])

data1.set_index('Price', inplace=True)
data2.set_index('Price', inplace=True)
data3.set_index('Price', inplace=True)
data4.set_index('Price', inplace=True)
data5.set_index('Price', inplace=True)

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

data1.rename(columns={'Close': 'S&P500'}, inplace=True)
data2.rename(columns={'Close': 'Nasdaq'}, inplace=True)
data3.rename(columns={'Close': 'Russell 2000'}, inplace=True)
data4.rename(columns={'Close': 'Dow Jones'}, inplace=True)
data5.rename(columns={'Close': 'US 10 Year Treasury'}, inplace=True)

df = pd.merge(data1, data2, on = "Date")
df = pd.merge(df, data3, on = "Date")
df = pd.merge(df, data4, on = "Date")
df = pd.merge(df, data5, on = "Date")

df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)
df['Date'] = df.index.date
df = df[df.index.year >= 2024]


df.to_excel('IndexReturn.xlsx', engine='openpyxl', index=False)