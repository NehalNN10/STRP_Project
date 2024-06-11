import pandas as pd
import matplotlib.pyplot as plt
import mplcursors as hover

rates = pd.read_csv('src/usdpkr_rates.csv')

mask = (
    (rates["date"].str.endswith("2024"))
    | (rates["date"].str.endswith("2023"))
    | (rates["date"].str.endswith("2022"))
    | (rates["date"].str.endswith("2021"))
)

rates = rates[mask]

rates = rates.iloc[::-1].reset_index(drop=True)

plt.figure(figsize=(10, 6))

plt.plot(rates['date'], rates['price'], color='green', linestyle='-')

n = 12
positions = rates.index[::n]
labels = rates["date"][::n]

# Remove the first 3 characters from each label
labels = [label[3:] for label in labels]

plt.xticks(positions, labels, rotation=45)
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('PKR')
plt.title('USD/PKR Exchange Rate')
plt.tight_layout()

# plt.gcf().canvas.set_window_title('USD/PKR Exchange Rate')

hover.cursor(hover=True)

plt.show()
