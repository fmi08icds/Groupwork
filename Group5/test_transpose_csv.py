import csv

# Pfad zur ursprünglichen .csv-Datei
input_file = './data/standard_test_data.csv'

# Pfad zur transponierten .csv-Datei
output_file = './data/transposed_data.csv'

# Lesen der ursprünglichen .csv-Datei
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Transponieren der Daten
transposed_data = list(map(list, zip(*data)))

# Schreiben der transponierten Daten in eine .csv-Datei
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(transposed_data)
