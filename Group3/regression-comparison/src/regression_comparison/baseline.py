from sklearn import linear_model, model_selection
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import logging

logger = logging.getLogger("regression_comparison")


## naive baseline for comparison:
## goal: have a list of models for comparison against self-implemented models
## requirements:
# 1. list of data
# 2. list of models
# 3. iterate through list of models and data, and return a list of scores
# 4. @todo: init a class with data, and return a list of scores


class RegressionBaseline:
    """
    A class for running baseline regressions on a given dataset.
    """

    def __init__(
        self, input_data: pd.DataFrame, dependent_index: str, name: str
    ) -> None:
        """
        Initialize the RegressionBaseline object.

        Args:
            data: A pandas DataFrame containing the data to run regressions on.
            dependent_index: The name of the dependent variable in the data.
            name: The name of the dataset.
        """

        ## list of models to benchmark / baseline for.
        self.models = [
            make_pipeline(StandardScaler(), linear_model.LinearRegression()),
            make_pipeline(StandardScaler(), linear_model.Ridge(alpha=0.5)),
            make_pipeline(StandardScaler(), linear_model.Lasso(alpha=0.1)),
            make_pipeline(StandardScaler(), linear_model.BayesianRidge()),
            make_pipeline(StandardScaler(), linear_model.SGDRegressor(random_state=0, max_iter=1000, tol=1e-3),),
        ]
        ## can be extended to other models

        # default test size
        self.test_size = 0.2
        # default random state for reproducibility (change to - None - for randomness)
        self.random_state = 0

        self.data_name = name
        self.dependent_index = dependent_index
        self.data = input_data
        self.data = self.preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.scores, self.mses = self.get_scores()
        self.scores = self.get_scores()[0]
        self.mses = self.get_scores()[1]

    def preprocess_data(self):
        """
        Preprocess the data by dropping rows with missing values and non-numeric/object columns
        such as datetimes, that need to be handled with other regression methods.

        Args:
            data: A pandas DataFrame containing the data to preprocess.

        Returns:
            The preprocessed pandas DataFrame.
        """
        self.data = self.data.dropna()
        for col in self.data.columns:
            logging.info(f"Checking {col} for non-numeric values.")
            if (
                self.data[col].dtype == "object"
                or self.data[col].dtype == "datetime64[ns]"
            ):
                logging.info(f"Removing {col} from data.")
                self.data = self.data.drop(col, axis=1)
        return self.data

    def split_data(self) -> "tuple[list, list, list, list]":
        """
        Split the data into training and testing sets.

        Returns:
            A tuple containing the training and testing sets for the independent and dependent variables.
        """
        X = self.data.drop(self.dependent_index, axis=1)
        y = self.data[self.dependent_index]
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X,
            y,
            test_size=self.test_size,
            train_size=1 - self.test_size,
            random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test

    def get_scores(self):
        """
        Create a list of regression models to run.

        Returns:
            A list of regression models.
        """
        scores = []
        mses = []
        for model in self.models:
            model.fit(self.X_train, self.y_train)
            score = model.score(self.X_test, self.y_test)
            scores.append(score)
            mse = metrics.mean_squared_error(self.y_test, model.predict(self.X_test))
            mses.append(mse)
        return scores, mses

    def get_results(self):
        """
        Gets the results of the regression.
        """
        return pd.DataFrame(
            {
                "Model": [
                    f"Model {self.models[i][1]}"
                    for i in range(
                        len(self.models)
                    )  ## toDO: fix if pipeline looks different it breaks.
                ],
                "Score": self.scores,
                "MSE": self.mses,
            }
        )

    def verbose_results(self):
        logger.info(f"Baseline Regression Models for {self.data_name}:")
        for i, model in enumerate(self.models):
            logger.info(f"Model {i+1}: {model}")
            logger.info(f"\nScores: {self.scores[i]}")
            logger.info(f"MSEs: {self.mses[i]}")


def run_baseline(datasets, dependent_variables):
    for dataset_name, dataset in datasets.items():
        dependent_variable = dependent_variables.get(dataset_name)
        logging.info(
            f"Running baseline regressions for {dataset_name} and variable {dependent_variable}."
        )
        baseline = RegressionBaseline(dataset, dependent_variable, name=dataset_name)
        baseline.verbose_results()
        return baseline.get_results()
