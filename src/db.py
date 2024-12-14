import pandas as pd

excel_file_path = "C:/Users/USER/Desktop/Personal/Bits/FibroPred-Challenge/data/bbddfpfcodificada.xlsx"

df = pd.read_excel(excel_file_path)

csv_file_path = "output.csv"

df.to_csv(csv_file_path, index=False)

print("Conversion completed. The CSV file is saved as", csv_file_path)
