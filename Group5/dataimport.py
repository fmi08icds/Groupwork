import pandas as pd
import matplotlib.pyplot as plt
import os  # operating system
import numpy as np


def get_test_data(labeled: bool, random=False):
    """
    Returns a pandas dataframe for testing
    :return: pandas dataframe containing test data
    """
    if labeled and not random:
        df = pd.read_csv("data/standard_test_data_labeled.csv", index_col=0)
    elif not labeled:
        df = pd.read_csv("data/standard_test_data.csv", index_col=0)
    else:
        create_random_sample(file_name="data/test_data_random.csv",
                             labeled=labeled)
        df = pd.read_csv("data/test_data_random.csv", index_col=0)
    return df


def data_preparation():
    # Task 1: Load data sets
    data_directory = "data"
    dataframe = pd.read_csv(os.path.join(data_directory, "data.csv"))
    print(dataframe.head(10))
    print(dataframe.describe())
    print(dataframe.shape)

    data_directory_labels = "data_labels"
    dataframe_labels = pd.read_csv(os.path.join(data_directory, "labels.csv"))
    print(dataframe_labels.head(20))
    print(dataframe_labels.describe())
    print(dataframe_labels.shape)

    # dataframe['cancer_type'] = dataframe_labels['Class']

    # Task 2: Merge the data sets

    dataframe = dataframe_labels.merge(dataframe, how='inner', on='Unnamed: 0')
    print(dataframe.head(10))
    print(dataframe.describe())
    print(dataframe.shape)

    print(dataframe.isnull().sum())

    # Task 3: Descriptive statistics of the data

    # Bar plot of the five different types of cancer

    dataframe['Class'].value_counts().plot(kind='bar', xlabel='cancer types',
                                           ylabel='frequency', title='Distribution of cancer types',
                                           color='green', figsize=(6, 7))
    plt.plot()
    plt.savefig('Distribution of cancer types.png')
    plt.show()

    # Relative distribution of the five different cancer types
    (dataframe['Class'].value_counts() / len(dataframe)).plot(kind='bar', xlabel='cancer types',
                                                              ylabel='frequency',
                                                              title='Relative distribution of cancer types',
                                                              color='green', figsize=(6, 7))

    plt.plot()
    plt.savefig('Relative distribution of cancer types.png')
    plt.show()

    # kernel density estimation (KDE) to estimate the probability density function of a
    # random variable
    # dataframe.groupby('Class')['gene_1'].plot.kde()
    # plt.plot()
    # plt.show()

    # plotting the density of the mean values of all genes
    fig, ax = plt.subplots(figsize=(8, 6))
    df = pd.DataFrame({'Class': dataframe['Class'],
                       'Density': dataframe.iloc[:, 2:].sum(axis=1) / len(dataframe.columns[2:])})
    labels = []
    for label, df_grouped in df.groupby('Class'):
        labels.append(label)
        df_grouped.plot(kind="kde", ax=ax)

    plt.legend(labels)
    plt.title('Density plot of the mean M of all the genes')
    plt.xlabel('Mean M')
    plt.plot()
    plt.savefig('Density plot of the mean M of all the genes.png')
    plt.show()


def get_df_merged_with_labels():
    # Task 1: Load data sets
    data_directory = "data"
    dataframe = pd.read_csv(os.path.join(data_directory, "data.csv"))

    dataframe_labels = pd.read_csv(os.path.join(data_directory, "labels.csv"))

    # Task 2: Merge the data sets
    dataframe = dataframe_labels.merge(dataframe, how='inner', on='Unnamed: 0')
    # Remove first column (Unnamed: 0)
    dataframe = dataframe.iloc[:, 1:]

    return dataframe


def create_random_sample(file_name="data/test_data_random.csv",
                         labeled=False, number_of_rows=5, number_of_genes=None):
    """
    Creates a random sample data set and saves it as csv.
    Names of the csv have to end with _random so that they are ignored
    (see .gitignore).
    :param file_name: name of the file (has to end with "_random.csv"
    :param labeled:
    :param number_of_rows:
    :param number_of_genes:
    :return: None
    """
    if not file_name[-11:] == "_random.csv":
        print("Please enter a valid file_name (has to end on '_random.csv'")
        file_name = "data/test_data_random.csv"

    df = get_df_merged_with_labels()
    if not number_of_genes:
        number_of_genes = df.shape[1]

    # TODO implement
    #
    # so in der Art:
    rand_sample = np.random.randint(low=0, high=df.shape[0], size=3)
    df = df.iloc[rand_sample, :]
    df.to_csv(file_name)
