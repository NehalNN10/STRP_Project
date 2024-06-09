import pandas as pd

# Load the data from the Excel file
input_file = 'DCOILBRENTEU.xlsx'
data = pd.read_excel(input_file)

# Function to convert date to biweekly format
def convert_to_biweekly(date):
    if date.day <= 15:
        return date.replace(day=1)
    else:
        return date.replace(day=15)

# Apply the conversion function to the 'date' column
data['date'] = pd.to_datetime(data['date']).apply(convert_to_biweekly)

# Remove duplicate dates
data = data.drop_duplicates(subset='date')

data['date'] = data['date'].dt.strftime('%d-%b-%Y')

# Save the biweekly data to a new Excel file
output_file = 'DCOILBRENTEU New.xlsx'
data.to_excel(output_file, index=False)

print("Biweekly data has been saved to", output_file)
