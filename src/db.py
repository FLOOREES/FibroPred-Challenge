import pandas as pd

excel_file_path = "data/FibroPredCODIFICADA.xlsx"

df = pd.read_excel(excel_file_path)

csv_file_path = "data/output.csv"

df.to_csv(csv_file_path, index=False)

print("Conversion completed. The CSV file is saved as", csv_file_path)
