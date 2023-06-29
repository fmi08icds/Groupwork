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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
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

    def preprocessing(self):
        # delete the duplicates rows
        self.df.drop_duplicates()

        """
            # Converting Categorical Variables Into Numeric Values as the follow:
            # gender
                0- Female
                1- Male
            -----------------------
            # Age
               Age <= 16 , then 0
               Age > 16 and Age <= 32 , then 1
               Age > 32 and Age Age <= 48 , then 2
               Age > 48 and Age <= 64 , then 3
               Age > 64 , then 4
            -----------------------
            # Smoking encoding
               0- No Info
               1- current
               2- not current
               3- former
               4- never
               5- ever
            -----------------------
            # BMI
               ( 0 - 18.5) ==> 0
               ( 18.5 - 25 ) ==> 1
               ( 25 - 30 ) ==> 2
               ( 30 - 35 ) ==> 3
               ( 35 - 40 ) ==> 4
               ( > 40 ) ==> 5
            -----------------------
            # HbA1c_level
               ( 0 - 5.6 ) ==> 0
               ( 5.6 - 6.4 ) ==> 1
               ( 6.4 - 6.9 ) ==> 2
               ( > 6.9 ) ==> 3
            -----------------------
            # blood_glucose_level
               ( 0 - 100 ) ==> 0
               ( 100 - 125 ) ==> 1
               ( 125 - 150 ) ==> 2
               ( 159 - 175 ) ==> 3
               ( > 175 ) ==> 4
        """

        # Gender
        gender_mapping = {'Male': 0, 'Female': 1}
        self.df['gender'] = self.df['gender'].map(gender_mapping)

        # Age
        self.df.loc[self.df['age'] <= 16, 'age'] = 0
        self.df.loc[(self.df['age'] > 16) & (self.df['age'] <= 32) , 'Age_band'] = 1
        self.df.loc[(self.df['age'] > 32) & (self.df['age'] <= 48), 'Age_band'] = 2
        self.df.loc[(self.df['age'] > 48) & (self.df['age'] <= 64), 'Age_band'] = 3
        self.df.loc[self.df['age'] > 64, 'age'] = 4

        # Smoking history
        mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
        self.df['smoking_history'] = self.df['smoking_history'].replace(mapping)

        # BMI
        bins = [0, 18.5, 25, 30, 35, 40, float('inf')]
        labels = [0, 1, 2, 3, 4, 5]
        self.df['bmi'] = pd.cut(self.df['bmi'], bins=bins, labels=labels)
        self.df['bmi'] = self.df['bmi'].astype(int)

        # HbA1c
        bins = [0, 5.6, 6.4, 6.9, float('inf')]
        labels = [0, 1, 2, 3]
        self.df['HbA1c_level'] = self.df['HbA1c_level'].astype(float)
        self.df['HbA1c_level'] = pd.cut(self.df['HbA1c_level'], bins=bins, labels=labels)

        # blood_glucose_level
        bins = [0, 100, 125, 150, 175, float('inf')]
        labels = [0, 1, 2, 3, 4]
        self.df['blood_glucose_level'] = self.df['blood_glucose_level'].astype(float)
        self.df['blood_glucose_level'] = pd.cut(self.df['blood_glucose_level'], bins=bins, labels=labels)

        # Convert the columns "hypertension" and "heart_disease" into a numerical one
        self.df['hypertension'] = self.df['hypertension'].astype(float)
        self.df['heart_disease'] = self.df['heart_disease'].astype(float)


        # Downsample the dataset
        df_majority = self.df[self.df['diabetes'] == 0]
        df_minority = self.df[self.df['diabetes'] == 1]
        df_majority_downsampled = resample(df_majority,
                                   replace=False,     # sampling without replacement
                                   n_samples=len(df_minority) * 3 ,    # match the size of minority class
                                   random_state=42)   # for reproducibility

        self.df = pd.concat([df_majority_downsampled, df_minority])

        return self.df




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
        X = self.df.drop('diabetes', axis=1)
        y = self.df['diabetes']
        X = X[['age','bmi','blood_glucose_level']]
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True,random_state=42)
        return X_train, X_test, y_train, y_test

#file = "/Users/abdulnaser/Desktop/Groupwork/Group2/data/diabetes_prediction_dataset.csv"

#--------------------#
