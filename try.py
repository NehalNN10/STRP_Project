import pandas as pd
import matplotlib.pyplot as plt

try_df = pd.read_csv('src/hsd_data.csv')
try_df_2 = pd.read_csv('src/petrol_data.csv')

# dropping extra fluctuations to ensure that the data changes every 15 days

# print(try_df)

mask = (
    (try_df["Period End"].str.startswith("01"))
    | (try_df["Period End"].str.startswith("16"))
)

mask_2 = (
    (try_df["Period End"].str.endswith("2024"))
    | (try_df["Period End"].str.endswith("2023"))
    | (try_df["Period End"].str.endswith("2022"))
    | (try_df["Period End"].str.endswith("2021"))
)
# mask = mask & mask_2

try_df = try_df[mask_2]
# try_df = try_df[mask]
try_df_2 = try_df_2[mask]

print(try_df, end="\n********************************************************\n")

# print(try_df_2)

# add missing dates into a text file
covered_dates = open('src/covered_dates.txt', 'w')

covered_dates.writelines("Here are the covered dates: \n\n")

for i in range(len(try_df)):
    print(try_df["Period End"].iloc[i])
    covered_dates.writelines(try_df["Period End"].iloc[i] + '\n')

covered_dates.close()

# ? Dataset is very inconsistent, will need to plug in missing values manually via reading articles
