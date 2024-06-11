import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from the Excel file
input_file = 'src/PKR _ US$ Exchange Rates.xlsx'
data = pd.read_excel(input_file)

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Filter the data for January and July
biannual_data = data[data['date'].dt.month.isin([1, 7])]
biannual_data = data[data['date'].dt.year.isin([2021, 2022, 2023, 2024])]
# Sort the data by date
biannual_data = biannual_data.sort_values(by='date')
biannual_data['price'] = biannual_data['price'].interpolate(method='linear')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(biannual_data['date'], biannual_data['price'], linestyle='-', color='b')

# Customize the plot
plt.title('PKR/USD Rate')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
#plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as an image
if os.path.exists('PKRUSD.png'):
    overwrite = input(f"{'PKRUSD.png'} already exists. Do you want to overwrite it? (yes/no): ")
    if overwrite.lower() != 'yes':
        print("File not overwritten.")
        plt.show()
    else:
        plt.savefig('PKRUSD.png')
        print(f"File {'PKRUSD.png'} overwritten.")
else:
    plt.savefig('PKRUSD.png')
    print(f"File {'PKRUSD.png'} saved.")

# Show the plot
plt.show()
