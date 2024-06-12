import pandas as pd
import matplotlib.pyplot as plt
import os
import mplcursors as hover

# Load data from the four Excel files
files = ['Diesel.xlsx', 'Petrol.xlsx', 'Brent.xlsx', 'USDPKR.xlsx']
labels = ['Diesel', 'Petrol', 'Brent Crude', 'Dollar']
dataframes = []

# Load, convert date, and apply cutoff for each dataframe
for file in files:
    df = pd.read_excel(file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2021-06-01']  # Apply cutoff
    dataframes.append(df)

# Set up the plot
plt.figure(figsize=(10, 6))

# Colors for the plots
colors = ['b', 'c', 'g', 'r']

# Plot each dataframe
for df, label, color in zip(dataframes, labels, colors):
    plt.plot(df['date'], df['price'], label=label, color=color, marker='o')

# for hovering
hover.cursor(hover=True)

# Customize the plot
plt.title('Price Trends')
plt.xlabel('Date')
plt.ylabel('Price (PKR)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot as an image
output_image = 'Price_Trends.png'
if os.path.exists(output_image):
    overwrite = input(f"{output_image} already exists. Do you want to overwrite it? (yes/no): ")
    if overwrite.lower() != 'yes':
        print("File not overwritten.")
        plt.show()
    else:
        plt.savefig(output_image)
        print(f"File {output_image} overwritten.")
else:
    plt.savefig(output_image)
    print(f"File {output_image} saved.")

# Show the plot
plt.show()
