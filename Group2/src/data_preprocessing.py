"""
The primary purpose of this file it to preprocess the diabetes dataset by
performing the following tasks:
1. Imputing null values
2. Handling outlier values
3. Conducting correlation analysis to remove irrelevant features
4. Engaging in feature engineering by inferring new features from existing ones.
5. Splitting the data into training, validation, and test datasets.
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataAnalyzer:
    def __init__(self, file):
        self.df = pd.read_csv(file, skipinitialspace=True, sep=',')

    def data_descriptions(self):
        print("name of columns of data")
        print(self.df.columns)
        print("---------------------------")
        print("---------------------------")
        print("Header of data")
        print(self.df.head())
        print("---------------------------")
        print("---------------------------")
        print("infomrmation about data ")
        print(self.df.info())
        print("---------------------------")
        print("---------------------------")
        print("nummber of th isnull vlaues ")
        print(self.df.isnull().sum())
        print("---------------------------")
        print("---------------------------")
        print("nummber of isna values" )
        print(self.df.isna().sum())
        print("---------------------------")
        print("---------------------------")
        print("statisical description of all the data ")
        print(self.df.describe())
        print("---------------------------")
        print("---------------------------")
        # columns counter the values 
        print("sum data depende on it values")
        col_liste = list(self.df.columns)
        for i in col_liste:
            print(self.df[i].value_counts())
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")

    def preprocessing_and_clean_data(self):
        # delete the duplicates rows
        self.df.drop_duplicates()

        """ 
            # Converting Categorical Variables Into Numeric Values Using LabelEncoder & OneHotEncoder
            Labelcodig as the follow:
            # gender
                0- Female
                1- Male   
            -----------------------    
            # Smoking encoding
                0- No Info
                1- current
                2- not current
                3- former
                4- never
                5- ever
        """
        X_cat = self.df.drop(['age', 'bmi'], axis=1)
        X_num = self.df[['age', 'bmi']]
        y = self.df['smoking_history']
        le = LabelEncoder()
        y = le.fit_transform(y)
        for col in X_cat.columns:
            X_cat[col] = le.fit_transform(X_cat[col])
        X = pd.concat([X_num, X_cat], axis=1)

        # Perform feature engineering

        # 0 : 'Underweight',18.5: 'Normal',24.9: 'Overweight',29.9: 'Obese'
        X['age_squared'] = X['age'] ** 2
        X['bmi_category'] = pd.cut(X['bmi'], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=[0, 18.5, 24.9, 29.9])
        X['age_bmi_interaction'] = X['age'] * X['bmi']
                
        print(X)
        return X

    


    def correlation(self):
        # calculate the correlation between all columns
        fig = plt.figure(figsize=(9, 9))
        sns.heatmap(self.df.corr(), annot=True)
        plt.show()
        return self.df.corr()
    

    def correlation_analysis(self):
        # Calculate the correlation matrix
        corr_matrix = self.df.corr()
        # Visualize the correlation matrix
        plt.figure(figsize=(9, 9))
        sns.heatmap(corr_matrix, annot=True)
        plt.title('Correlation Analysis')
        plt.show()

        # Identify features with low correlation
        threshold = 0.2
        low_corr_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) < threshold:
                    colname_i = corr_matrix.columns[i]
                    colname_j = corr_matrix.columns[j]
                    low_corr_features.add(colname_i)
                    low_corr_features.add(colname_j)

        # Remove features with low correlation
        self.df.drop(low_corr_features, axis=1, inplace=True)


    def data_split(self):
        # Preprocess and clean the data
        age_group_tables = self.preprocessing_and_clean_data()
        
        # Create an empty dictionary to store the split datasets
        split_datasets = {}

                # Split the data for each age group
        for age_group, data in age_group_tables.items():
            # Convert 'data' to DataFrame if it is a Series
            if isinstance(data, pd.Series):
                data = pd.DataFrame(data)

            # Check if 'diabetes' column exists
            if 'diabetes' not in data.columns:
                print(f"Skipping data splitting for age group '{age_group}' as 'diabetes' column is not present.")
                continue

            # Separate features and target variable
            X = data.drop('diabetes', axis=1)
            y = data['diabetes']

            # Split the data into training, validation, and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

            # Store the split datasets for the age group
            split_datasets[age_group] = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }

        # Print the shapes of the resulting datasets for each age group
        for age_group, datasets in split_datasets.items():
            print(f"Age Group: {age_group}")
            print("Training set shape:", datasets['X_train'].shape, datasets['y_train'].shape)
            print("Validation set shape:", datasets['X_val'].shape, datasets['y_val'].shape)
            print("Test set shape:", datasets['X_test'].shape, datasets['y_test'].shape)
            print("---------------------------")
        print(split_datasets)
        # Return the split datasets
        return split_datasets
file = "Group2\data\diabetes_prediction_dataset.csv"  
analyzer = DataAnalyzer(file)

# analyzer.data_descriptions()
# print(analyzer.preprocessing_and_clean_data())
# analyzer.correlation_analysis()
# analyzer.correlation()
analyzer.data_split()
#--------------------#
