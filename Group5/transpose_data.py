import csv

# Path of the original .csv-File
input_file = './data/standard_test_data.csv'

# Path to the transposed .csv-File
output_file = './data/transposed_data.csv'

# Reading the original .csv-File
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Transpose the data
transposed_data = list(map(list, zip(*data)))

# Writing the transposed File in a .csv-File
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(transposed_data)
