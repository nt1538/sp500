import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('new_S&P500-raw_prices.csv')
data1 = data1[['Price', 'Close']]
data1 = data1.iloc[2:]
data1['Price'] = pd.to_datetime(data1['Price'])
data1.set_index('Price', inplace=True)
full_date_range = pd.date_range(start=data1.index.min(), end=data1.index.max(), freq='D')
data1 = data1.reindex(full_date_range)
data1.ffill(inplace=True)
data1.reset_index(inplace=True)
data1.rename(columns={'index': 'Date'}, inplace=True)
data1.rename(columns={'Close': 'S&P500'}, inplace=True)

data1['S&P500'] = pd.to_numeric(data1['S&P500'], errors='coerce')

# drop_thresholds = [0.127]  # 18.9% and 12.7%
# max_lookahead = 90
# min_gap_days = 30

# # Detect drop dates
# drop_dates = []

# for i in range(len(data1)):
#     start_price = data1.loc[i, 'S&P500']
#     if pd.isna(start_price):
#         continue
#     for j in range(1, max_lookahead + 1):
#         if i + j >= len(data1):
#             break
#         end_price = data1.loc[i + j, 'S&P500']
#         if pd.isna(end_price):
#             continue
#         drop_pct = (start_price - end_price) / start_price
#         for threshold in drop_thresholds:
#             if drop_pct >= threshold:
#                 drop_date = data1.loc[i + j, 'Date']
#                 drop_dates.append(drop_date)
#                 break

# drop_dates = sorted(set(drop_dates))

# # Filter to keep only last one in close periods
# filtered_dates = []
# for date in drop_dates:
#     if not filtered_dates or (date - filtered_dates[-1]).days > min_gap_days:
#         filtered_dates.append(date)
#     else:
#         filtered_dates[-1] = date

# # Calculate 6-year increase rates
# results = []
# plot_data = []

# data1.set_index('Date', inplace=True)

# for date in filtered_dates:
#     if date not in data1.index:
#         continue
#     end_date = date + pd.DateOffset(years=6)
#     if end_date not in data1.index:
#         continue
#     start_price = data1.loc[date, 'S&P500']
#     end_price = data1.loc[end_date, 'S&P500']
#     increase_rate = (end_price - start_price) / start_price
#     results.append({
#         'Drop Date': date,
#         '6Y Later Date': end_date,
#         'Start Price': start_price,
#         'End Price': end_price,
#         'Increase Rate (%)': round(increase_rate * 100, 2)
#     })

#     # Extract time series for plotting
#     time_series = data1.loc[date:end_date]['S&P500']
#     plot_data.append((date, time_series))

# # Plotting
# plt.figure(figsize=(12, 8))
# for drop_date, series in plot_data:
#     series = series / series.iloc[0] * 100 - 100 # Normalize
#     plt.plot(series.index, series.values, label=drop_date.strftime('%Y-%m-%d'))

# plt.title('S&P500 Growth After Major Drops (Next 6 Years)')
# plt.ylabel('Normalized S&P500')
# plt.xlabel('Date')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

drop_thresholds = [0.127]
max_lookahead = 90
min_gap_days = 30

# Detect drop dates
drop_dates = []

for i in range(len(data1)):
    start_price = data1.loc[i, 'S&P500']
    if pd.isna(start_price):
        continue
    for j in range(1, max_lookahead + 1):
        if i + j >= len(data1):
            break
        end_price = data1.loc[i + j, 'S&P500']
        if pd.isna(end_price):
            continue
        drop_pct = (start_price - end_price) / start_price
        for threshold in drop_thresholds:
            if drop_pct >= threshold:
                drop_date = data1.loc[i + j, 'Date']
                drop_dates.append(drop_date)
                break

drop_dates = sorted(set(drop_dates))

# Filter to keep only last one in close periods
filtered_dates = []
for date in drop_dates:
    if not filtered_dates or (date - filtered_dates[-1]).days > min_gap_days:
        filtered_dates.append(date)
    else:
        filtered_dates[-1] = date

# Calculate 6-year increase rates
results = []

data1.set_index('Date', inplace=True)

for date in filtered_dates:
    if date not in data1.index:
        continue

    window_start = date - pd.Timedelta(days=90)
    if window_start < data1.index.min():
        continue
    window_data = data1.loc[window_start:date]['S&P500']
    if len(window_data) < 2:
        continue
    max_drop = max((window_data.iloc[0] - window_data) / window_data.iloc[0])

    end_date = date + pd.DateOffset(years=6)
    if end_date not in data1.index:
        continue
    start_price = data1.loc[date, 'S&P500']
    end_price = data1.loc[end_date, 'S&P500']
    increase_rate = (end_price - start_price) / start_price

    results.append({
        'Drop Date': date.date(),
        'Max Drop in Previous 90 Days (%)': round(max_drop * 100, 2),
        'Start Price': round(start_price, 2),
        '6Y Later Price': round(end_price, 2),
        '6Y Increase Rate (%)': round(increase_rate * 100, 2)
    })

# Output to Excel
output_df = pd.DataFrame(results)
output_df.to_excel('SP500_Drop_and_6Y_Return.xlsx', engine='openpyxl', index=False)