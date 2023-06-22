import our_pca
import dataimport
from data_cleaning import *


# Get a random DataFrame
#   with nr_r rows and nr_c columns (both will be set by user-input)
print("Start by importing a random sample of the whole dataset..")
nr_r = input("How many rows (individuals) do you want in your sample?")
nr_c = input("How many columns (attributes) do you want in your sample?")
try:
    nr_r = int(nr_r)
    nr_c = int(nr_c)
    print("Please wait..")
except ValueError:
    nr_r = 300
    nr_c = 2000
    print("The input was not valid! ..continue with nr_rows=300, nr_cols=2000")

# Get the random DataFrame without labels
df = dataimport.get_random_sample(labeled=False, nr_rows=nr_r, nr_cols=nr_c)

# Using a labeled DataFrame works like this:
#
# df_labeled = dataimport.get_random_sample(labeled=True, nr_rows=nr_r,
#                                           nr_cols=nr_c)
# print("Remove labels for the application")
# df = df_labeled.iloc[:, :-1].copy()
# print("DataFrame without labels:\n", df)

print("Preprocessing..")
clean_data = preprocessing(df)
print("cleaned data:\n", clean_data)

# Ask the user how many components she wants to extract
n_components = input("How many components do you want the PCA to find?")
try:
    n_components = int(n_components)
    print("Okay")
except ValueError:
    n_components = 2
    print("The input was not valid! ..continue with n_components=2")

# Check the numbers for validity
if n_components > nr_r:
    n_components = nr_r
    print("n_components is too big! ..continue with n_components=nr_rows")

print(f"PCA with {n_components} components..")

# Get eigenvectors
eigenvalues, eigenvectors = our_pca.our_pca(clean_data, n_components)

print("eigenvalues:\n", eigenvalues)
print("eigenvectors with shape", eigenvectors.shape)
for ev in eigenvectors:
    print(ev)

# Apply the pca-components to data
print("Apply components to data (projection)..")
projected_df = our_pca.apply_components(clean_data, eigenvectors)
print("Projected data:\n", projected_df)
print("FIN")
