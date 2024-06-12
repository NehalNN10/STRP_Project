import pandas as pd
import matplotlib.pyplot as plt
import mplcursors as hover
import numpy as np

rates = pd.read_csv('src/usdpkr_rates.csv')

mask = (
    (rates["date"].str.endswith("2024"))
    | (rates["date"].str.endswith("2023"))
    | (rates["date"].str.endswith("2022"))
    | (rates["date"].str.endswith("2021"))
)

rates = rates[mask]

rates = rates.iloc[::-1].reset_index(drop=True)

# plt.figure(figsize=(10, 6))

# plt.plot(rates['date'], rates['price'], color='green', linestyle='-')

# n = 12
# positions = rates.index[::n]
# labels = rates["date"][::n]

# # Remove the first 3 characters from each label
# labels = ['-'.join(label[3:].split('-')[::-1]) for label in labels]

# plt.xticks(positions, labels, rotation=45)
# plt.grid(True)
# plt.xlabel('Date')
# plt.ylabel('PKR')
# plt.title('USD/PKR Exchange Rate')
# plt.tight_layout()

# # plt.gcf().canvas.set_window_title('USD/PKR Exchange Rate')

# hover.cursor(hover=True)

# plt.show()

# plot brent oil rates but in pkr instead

# brent_rate = pd.read_csv('src/brent_crude_oil_rates_usd.csv')
brent_rate = pd.read_csv('src/brent_oil_rates.csv')

mask = (brent_rate["Date"].str.endswith("01")) | (
    brent_rate["Date"].str.endswith("15")
)

mask_2 = (
    (brent_rate["Date"].str.startswith("2024"))
    | (brent_rate["Date"].str.startswith("2023"))
    | (brent_rate["Date"].str.startswith("2022"))
    | (brent_rate["Date"].str.startswith("2021"))
)

brent_rate = brent_rate[mask & mask_2]

brent_rate = brent_rate.reset_index(drop=True)

brent_rate['usd_pkr_rate'] = rates['price']
brent_rate['price_in_pkr'] = brent_rate['Price'] * brent_rate['usd_pkr_rate']

# number of litres in a barrel of crude oil
litres = 158.987295

brent_rate['litre_price'] = brent_rate['price_in_pkr'] / litres

# print(brent_rate)
plt.figure(figsize=(10, 6))

# align both graphs
brent_rate["Date"] = pd.to_datetime(brent_rate["Date"])
rates["date"] = pd.to_datetime(rates["date"])

# Set 'Date' columns as index
brent_rate.set_index("Date", inplace=True)
rates.set_index("date", inplace=True)


plt.plot(rates["price"], color="green", linestyle="-", marker='o', label='USD/PKR Exchange Rate')
plt.plot(
    brent_rate["litre_price"],
    color="blue",
    linestyle="-",
    marker="o",
    label="Crude Oil Price",
)

n = 12
positions = np.arange(0, len(brent_rate), n)
labels = brent_rate.index[positions].strftime("%Y-%m-%d")

# print(positions)
# print(labels)
# print(rates.index)
if input() == '':
    print('Booyah!')

# Select every nth date from brent_rate.index
dates = brent_rate.index[::n]

plt.xticks(dates, dates.strftime("%Y-%m-%d"), rotation=45)

# plt.xticks(positions, labels, rotation=45)
plt.xlabel('Date')
plt.ylabel('PKR')
plt.title('Brent Crude Oil Price in PKR per Litre')
plt.grid(True)
plt.legend()
plt.tight_layout()

hover.cursor(hover=True)

plt.show()
