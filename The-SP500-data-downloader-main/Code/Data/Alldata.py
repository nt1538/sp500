import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
data1 = pd.read_csv('new_S&P500-raw_prices.csv')
data2 = pd.read_csv('new_Russell 2000-raw_prices.csv')
data3 = pd.read_csv('new_Nasdaq-raw_prices.csv')
data4 = pd.read_csv('new_MSCI KLD 400-raw_prices.csv')
data5 = pd.read_csv('new_MSCI Emerging Markets-raw_prices.csv')
data6 = pd.read_csv('new_EURO STOXX 50-raw_prices.csv')
data7 = pd.read_csv('new_Dow-Jones-raw_prices.csv')
data8 = pd.read_csv('new_MSCI EAFE-raw_prices.csv')
data9 = pd.read_csv('US 10 year Treasury-raw_prices.csv')
data10 = pd.read_csv('US 5 year Treasury-raw_prices.csv')
data11 = pd.read_csv('Communication Services-raw_prices.csv')
data12 = pd.read_csv('Consumer Discretionary-raw_prices.csv')
data13 = pd.read_csv('Consumer Staples-raw_prices.csv')
data14 = pd.read_csv('Energy-raw_prices.csv')
data15 = pd.read_csv('Financials-raw_prices.csv')
data16 = pd.read_csv('Health Care-raw_prices.csv')
data17 = pd.read_csv('Industrials-raw_prices.csv')
data18 = pd.read_csv('Materials-raw_prices.csv')
data19 = pd.read_csv('Real Estate-raw_prices.csv')
data20 = pd.read_csv('Technology-raw_prices.csv')
data21 = pd.read_csv('Utilities-raw_prices.csv')

data1 = data1[['Date', 'Adj Close']]
data2 = data2[['Date', 'Adj Close']]
data3 = data3[['Date', 'Adj Close']]
data4 = data4[['Date', 'Adj Close']]
data5 = data5[['Date', 'Adj Close']]
data6 = data6[['Date', 'Adj Close']]
data7 = data7[['Date', 'Adj Close']]
data8 = data8[['Date', 'Adj Close']]
data9 = data9[['Date', 'Adj Close']]
data10 = data10[['Date', 'Adj Close']]
data11 = data11[['Date', 'Adj Close']]
data12 = data12[['Date', 'Adj Close']]
data13 = data13[['Date', 'Adj Close']]
data14 = data14[['Date', 'Adj Close']]
data15 = data15[['Date', 'Adj Close']]
data16 = data16[['Date', 'Adj Close']]
data17 = data17[['Date', 'Adj Close']]
data18 = data18[['Date', 'Adj Close']]
data19 = data19[['Date', 'Adj Close']]
data20 = data20[['Date', 'Adj Close']]
data21 = data21[['Date', 'Adj Close']]

data1['Date'] = pd.to_datetime(data1['Date'])
data2['Date'] = pd.to_datetime(data2['Date'])
data3['Date'] = pd.to_datetime(data3['Date'])
data4['Date'] = pd.to_datetime(data4['Date'])
data5['Date'] = pd.to_datetime(data5['Date'])
data6['Date'] = pd.to_datetime(data6['Date'])
data7['Date'] = pd.to_datetime(data7['Date'])
data8['Date'] = pd.to_datetime(data8['Date'])
data9['Date'] = pd.to_datetime(data9['Date'])
data10['Date'] = pd.to_datetime(data10['Date'])
data11['Date'] = pd.to_datetime(data11['Date'])
data12['Date'] = pd.to_datetime(data12['Date'])
data13['Date'] = pd.to_datetime(data13['Date'])
data14['Date'] = pd.to_datetime(data14['Date'])
data15['Date'] = pd.to_datetime(data15['Date'])
data16['Date'] = pd.to_datetime(data16['Date'])
data17['Date'] = pd.to_datetime(data17['Date'])
data18['Date'] = pd.to_datetime(data18['Date'])
data19['Date'] = pd.to_datetime(data19['Date'])
data20['Date'] = pd.to_datetime(data20['Date'])
data21['Date'] = pd.to_datetime(data21['Date'])

data1.set_index('Date', inplace=True)
data2.set_index('Date', inplace=True)
data3.set_index('Date', inplace=True)
data4.set_index('Date', inplace=True)
data5.set_index('Date', inplace=True)
data6.set_index('Date', inplace=True)
data7.set_index('Date', inplace=True)
data8.set_index('Date', inplace=True)
data9.set_index('Date', inplace=True)
data10.set_index('Date', inplace=True)
data11.set_index('Date', inplace=True)
data12.set_index('Date', inplace=True)
data13.set_index('Date', inplace=True)
data14.set_index('Date', inplace=True)
data15.set_index('Date', inplace=True)
data16.set_index('Date', inplace=True)
data17.set_index('Date', inplace=True)
data18.set_index('Date', inplace=True)
data19.set_index('Date', inplace=True)
data20.set_index('Date', inplace=True)
data21.set_index('Date', inplace=True)

full_date_range = pd.date_range(start=data1.index.min(), end=data1.index.max(), freq='D')

data1 = data1.reindex(full_date_range)
data2 = data2.reindex(full_date_range)
data3 = data3.reindex(full_date_range)
data4 = data4.reindex(full_date_range)
data5 = data5.reindex(full_date_range)
data6 = data6.reindex(full_date_range)
data7 = data7.reindex(full_date_range)
data8 = data8.reindex(full_date_range)
data9 = data9.reindex(full_date_range)
data10 = data10.reindex(full_date_range)
data11 = data11.reindex(full_date_range)
data12 = data12.reindex(full_date_range)
data13 = data13.reindex(full_date_range)
data14 = data14.reindex(full_date_range)
data15 = data15.reindex(full_date_range)
data16 = data16.reindex(full_date_range)
data17 = data17.reindex(full_date_range)
data18 = data18.reindex(full_date_range)
data19 = data19.reindex(full_date_range)
data20 = data20.reindex(full_date_range)
data21 = data21.reindex(full_date_range)

data1.ffill(inplace=True)
data2.ffill(inplace=True)
data3.ffill(inplace=True)
data4.ffill(inplace=True)
data5.ffill(inplace=True)
data6.ffill(inplace=True)
data7.ffill(inplace=True)
data8.ffill(inplace=True)
data9.ffill(inplace=True)
data10.ffill(inplace=True)
data11.ffill(inplace=True)
data12.ffill(inplace=True)
data13.ffill(inplace=True)
data14.ffill(inplace=True)
data15.ffill(inplace=True)
data16.ffill(inplace=True)
data17.ffill(inplace=True)
data18.ffill(inplace=True)
data19.ffill(inplace=True)
data20.ffill(inplace=True)
data21.ffill(inplace=True)

data1.reset_index(inplace=True)
data2.reset_index(inplace=True)
data3.reset_index(inplace=True)
data4.reset_index(inplace=True)
data5.reset_index(inplace=True)
data6.reset_index(inplace=True)
data7.reset_index(inplace=True)
data8.reset_index(inplace=True)
data9.reset_index(inplace=True)
data10.reset_index(inplace=True)
data11.reset_index(inplace=True)
data12.reset_index(inplace=True)
data13.reset_index(inplace=True)
data14.reset_index(inplace=True)
data15.reset_index(inplace=True)
data16.reset_index(inplace=True)
data17.reset_index(inplace=True)
data18.reset_index(inplace=True)
data19.reset_index(inplace=True)
data20.reset_index(inplace=True)
data21.reset_index(inplace=True)

data1.rename(columns={'index': 'Date'}, inplace=True)
data2.rename(columns={'index': 'Date'}, inplace=True)
data3.rename(columns={'index': 'Date'}, inplace=True)
data4.rename(columns={'index': 'Date'}, inplace=True)
data5.rename(columns={'index': 'Date'}, inplace=True)
data6.rename(columns={'index': 'Date'}, inplace=True)
data7.rename(columns={'index': 'Date'}, inplace=True)
data8.rename(columns={'index': 'Date'}, inplace=True)
data9.rename(columns={'index': 'Date'}, inplace=True)
data10.rename(columns={'index': 'Date'}, inplace=True)
data11.rename(columns={'index': 'Date'}, inplace=True)
data12.rename(columns={'index': 'Date'}, inplace=True)
data13.rename(columns={'index': 'Date'}, inplace=True)
data14.rename(columns={'index': 'Date'}, inplace=True)
data15.rename(columns={'index': 'Date'}, inplace=True)
data16.rename(columns={'index': 'Date'}, inplace=True)
data17.rename(columns={'index': 'Date'}, inplace=True)
data18.rename(columns={'index': 'Date'}, inplace=True)
data19.rename(columns={'index': 'Date'}, inplace=True)
data20.rename(columns={'index': 'Date'}, inplace=True)
data21.rename(columns={'index': 'Date'}, inplace=True)

# data1['Adj Close'] = data1['Adj Close'].fillna(0)
# data2['Adj Close'] = data2['Adj Close'].fillna(0)
# data3['Adj Close'] = data3['Adj Close'].fillna(0)
# data4['Adj Close'] = data4['Adj Close'].fillna(0)
# data5['Adj Close'] = data5['Adj Close'].fillna(0)
# data6['Adj Close'] = data6['Adj Close'].fillna(0)
# data7['Adj Close'] = data7['Adj Close'].fillna(0)

data1.rename(columns={'Adj Close': 'S&P500'}, inplace=True)
data2.rename(columns={'Adj Close': 'Russell 2000'}, inplace=True)
data3.rename(columns={'Adj Close': 'Nasdaq'}, inplace=True)
data4.rename(columns={'Adj Close': 'MSCI KLD 400'}, inplace=True)
data5.rename(columns={'Adj Close': 'MSCI Emerging Markets'}, inplace=True)
data6.rename(columns={'Adj Close': 'EURO STOXX 50'}, inplace=True)
data7.rename(columns={'Adj Close': 'Dow Jones'}, inplace=True)
data8.rename(columns={'Adj Close': 'MSCI EAFE'}, inplace=True)
data9.rename(columns={'Adj Close': 'US 10 Year Treasury'}, inplace=True)
data10.rename(columns={'Adj Close': 'US 5 Year Treasury'}, inplace=True)
data11.rename(columns={'Adj Close': 'Communication Services'}, inplace=True)
data12.rename(columns={'Adj Close': 'Consumer Discretionary'}, inplace=True)
data13.rename(columns={'Adj Close': 'Consumer Staples'}, inplace=True)
data14.rename(columns={'Adj Close': 'Energy'}, inplace=True)
data15.rename(columns={'Adj Close': 'Finalcials'}, inplace=True)
data16.rename(columns={'Adj Close': 'Health Care'}, inplace=True)
data17.rename(columns={'Adj Close': 'Industrials'}, inplace=True)
data18.rename(columns={'Adj Close': 'Materials'}, inplace=True)
data19.rename(columns={'Adj Close': 'Real Estate'}, inplace=True)
data20.rename(columns={'Adj Close': 'Technology'}, inplace=True)
data21.rename(columns={'Adj Close': 'Utilities'}, inplace=True)

df = pd.merge(data1, data3, on = "Date")
df = pd.merge(df, data2, on = "Date")
df = pd.merge(df, data7, on = "Date")
df = pd.merge(df, data8, on = "Date")
df = pd.merge(df, data5, on = "Date")
df = pd.merge(df, data4, on = "Date")
df = pd.merge(df, data6, on = "Date")
df = pd.merge(df, data9, on = "Date")
df = pd.merge(df, data10, on = "Date")
df = pd.merge(df, data11, on = "Date")
df = pd.merge(df, data12, on = "Date")
df = pd.merge(df, data13, on = "Date")
df = pd.merge(df, data14, on = "Date")
df = pd.merge(df, data15, on = "Date")
df = pd.merge(df, data16, on = "Date")
df = pd.merge(df, data17, on = "Date")
df = pd.merge(df, data18, on = "Date")
df = pd.merge(df, data19, on = "Date")
df = pd.merge(df, data20, on = "Date")
df = pd.merge(df, data21, on = "Date")


#df = pd.DataFrame([data1["Date"], data1['S&P500'], data3['Nasdaq'],data2['Russell 2000'],data6['MSCI EAFE'],
#      data5['MSCI Emerging Markets'],data4['MSCI KLD 400'],data7['Dow Jones']])

df.to_csv('data compares to sp500.csv', index=False)

# for val in list:
#     price_list.append(val)

# for i in range(1,len-1):
#     start = price_list[i-1]
#     end = price_list[i]
#     inc = end / start - 1
#     if inc < 0: 
#         inc = 0
#     elif inc > 0.10:
#         inc = 0.10
#     curVal = curVal * (1 + inc)
#     IULPrice.append(curVal)

# for i in range(24,len-1):
#     start = price_list[i-24]
#     end = price_list[i]
#     startIUL = IULPrice[i-24]
#     endIUL = IULPrice[i]
#     endYear = startYear + 24
#     yearList.append(f"{startYear}-{endYear}")
#     CAGR.append(((end/start) ** (1/24) - 1) * 100)
#     IUL.append(((endIUL/startIUL) ** (1/24) - 1) * 100)
#     startYear = startYear+1

# plt.figure(figsize=(14, 7))
# plt.plot(yearList, CAGR, label='S&P 500 Index Value')
# plt.plot(yearList, IUL, label='IUL Value')
# plt.xlabel('Year')
# plt.ylabel('Value')
# plt.title('Comparison of S&P 500 CAGR and IUL Return')
# plt.xticks(rotation=90)
# plt.legend()
# plt.grid(True)
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
# plt.ylim(0,16)
# plt.tight_layout()  # Adjust layout to make room for x-axis labels
# plt.show()
