import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from csv file
data = pd.read_csv('src/petrol_data.csv')
hsd_data = pd.read_csv('src/hsd_data.csv')

a = data.columns[1:-1]
data = data.drop(columns=a, inplace=False)
hsd_data = hsd_data.drop(columns=a, inplace=False)

data = data.rename(columns={'Period End':'Date', 'Pakistan':'Price'}, inplace=False)
hsd_data = hsd_data.rename(columns={'Period End':'Date', 'Pakistan':'Price'}, inplace=False)

data = data.drop(index=0).reset_index(drop=True)
hsd_data = hsd_data.drop(index=0).reset_index(drop=True)

data["Date"] = pd.to_datetime(data["Date"])
hsd_data["Date"] = pd.to_datetime(hsd_data["Date"])

data["Price"] = pd.to_numeric(data["Price"]) # * necessary otherwise graph constantly rises
hsd_data["Price"] = pd.to_numeric(hsd_data["Price"]) # * necessary otherwise graph constantly rises

mask = (
    (data["Date"].dt.year == 2024)
    | (data["Date"].dt.year == 2023)
    | (data["Date"].dt.year == 2022)
    | (data["Date"].dt.year == 2021)
)

mask_2 = (
    (hsd_data["Date"].dt.year == 2024)
    | (hsd_data["Date"].dt.year == 2023)
    | (hsd_data["Date"].dt.year == 2022)
    | (hsd_data["Date"].dt.year == 2021)
)

data = data[mask]
hsd_data = hsd_data[mask]

# ? clean data to allow for plotting of pkr/usd exchange rate

def get_nearest_fortnight(date):
    if 1 < date.day <= 15:
        return date.replace(day=15)
    elif date.day != 1:
        return date.replace(day=1, month=date.month+1 if date.month < 12 else 1, year=date.year+1 if date.month == 12 else date.year)

def biweekly(df):
    df["near"] = df["Date"].apply(get_nearest_fortnight)
    df = df.drop_duplicates(subset="near", keep='last').reset_index(drop=True)
    return df

data = biweekly(data)
hsd_data = biweekly(hsd_data)

data = data.sort_values(by="near").reset_index(drop=True)
hsd_data = hsd_data.sort_values(by="near").reset_index(drop=True)

# for index, row in data.iterrows():
    # print(f'{row["Date"]}: {row["near"]}: {row["Price"]}', end="\n")

# input("Joe mama")

# * handling the exchange rate dataframe

usd_pkr = pd.read_csv('src/usdpkr_rates.csv')
usd_pkr = usd_pkr.rename(columns={"date": "Date", "price": "Price"}, inplace=False)
usd_pkr["Date"] = pd.to_datetime(usd_pkr["Date"])

mask_3 = (
    (usd_pkr["Date"].dt.year == 2024)
    | (usd_pkr["Date"].dt.year == 2023)
    | (usd_pkr["Date"].dt.year == 2022)
    | (usd_pkr["Date"].dt.year == 2021)
)

usd_pkr = usd_pkr[mask_3]
usd_pkr = usd_pkr.sort_values(by="Date").reset_index(drop=True)

# * the actual plotting

plt.figure(figsize=(10, 6))
plt.plot(data["near"], data['Price'], linestyle='-', color='b', label='Petrol', marker='*')
plt.plot(hsd_data["near"], hsd_data['Price'], linestyle='-', color='r', label='HSD')
plt.plot(usd_pkr["Date"], usd_pkr['Price'], linestyle='-', color='g', label='USD/PKR exchange rate')
plt.title('Fuel prices in PKR')
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('PKR')
plt.legend()
plt.tight_layout()

plt.show()

# TODO: look at graph and notice sidewards spikes
