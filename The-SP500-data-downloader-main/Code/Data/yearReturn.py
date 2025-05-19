import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def yearReturn(data, start_year, end_year):

    startyear = start_year
    # Set index and convert date column to datetime
    data.set_index('Date', inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Date'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])

    RTN = []
    yearList = []

    while start_year <= end_year:
        CurYearData = data[data['Date'].dt.year == start_year]
        CurYearData.reset_index(inplace=True)
        CurYearData = CurYearData['Adj Close']

        inc = (CurYearData[len(CurYearData) - 1] - CurYearData[0]) / CurYearData[0]
        
        RTN.append(inc * 100)
        yearList.append(f"{start_year}")
        start_year += 1

    start_year = startyear

    # Plot results
    plt.figure(figsize=(14, 7))

    ymax = np.max(RTN)
    ymin = np.min(RTN)
    comp = (ymax - ymin)/20
    avg_RTN = np.mean(RTN)

    plt.title(f'S&P500 Yearly Growth')
    plt.plot(yearList, RTN, label=f'S&P500 Yearly Growth')
    plt.text(yearList[-1], ymin + comp, f'Average S&P500 Annual Return: {avg_RTN:.2f}%', color='blue', ha='right', va='bottom', fontsize=10)
    for i, txt in enumerate(RTN):
        plt.text(yearList[i], RTN[i], f'{txt:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.ylim(ymin, ymax)
    plt.tight_layout()  # Adjust layout to make room for x-axis labels
    plt.show()

    newData = {
        "Year List": yearList,
        f"S&P500 Yearly Growth": RTN
    }
    df = pd.DataFrame(newData)
    df.to_excel('YearReturn.xlsx', engine='openpyxl', index=False)


sp500data = pd.read_csv('sp500_data_with_weekends.csv')
yearReturn(sp500data, 1957,2023)
