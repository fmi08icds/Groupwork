"""This is our interactive app to learn about """
from enum import Enum

import localreg
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn import linear_model, metrics

from regression_edu.models.linear_regression import LinearRegression
from regression_edu.models.locally_weighted_regression import \
    LocallyWeightedRegression

# initialise dash app
app = Dash(__name__)

#  read datasets
df1 = pd.read_excel(
    io="../data/realworld/prostate.xlsx", usecols=lambda x: "Unnamed: 0" not in x
)
df2 = pd.read_csv(
    filepath_or_buffer="../data/realworld/winequality-red.csv", delimiter=";"
)
df3 = pd.read_excel(io="../data/realworld/real_estate.xlsx")


class Dataset(Enum):
    """
    This enum models the different options of datasets that are available for the
    comparison.
    """

    PROSTATE = "Prostate"
    WINE_QUALITY = "Wine Quality"
    REAL_ESTATE = "RealEstate"


dataset_config = {
    Dataset.PROSTATE: {
        "dependent_variable": df1[df1.columns[-2]],
        "training_data": df1.drop(df1.columns[-2], axis=1),
    },
    Dataset.WINE_QUALITY: {
        "dependent_variable": df2[df2.columns[-1]],
        "training_data": df2.drop(df2.columns[-1], axis=1),
    },
    Dataset.REAL_ESTATE: {
        "dependent_variable": df3[df3.columns[-1]],
        "training_data": df3.drop(df3.columns[-1], axis=1),
    },
}


# create dropdown element for selecting the dataset
dataset_dropdown = dcc.Dropdown(
    [dataset.value for dataset in Dataset],
    clearable=False,
    placeholder="Select a dataset",
    id="dataset-dropdown",
    value="Prostate",
)  # set default value to prostate dataset

# create dropdown element 1 for selecting the 1st independent variable
indep_dropdown_1 = dcc.Dropdown(
    options=df1.columns,
    clearable=False,
    placeholder="Select an independent variable",
    id="indep-dropdown-1",
    value=df1.columns[1],
)


# create a callback relationship linking database-dropdown to indep-dropdown
@app.callback(
    Output("indep-dropdown-1", "options"),
    Output("indep-dropdown-1", "value"),
    Input("dataset-dropdown", "value"),
)
def update_indep_dropdown_1(input_dataset):
    """Updates the dropdown of the independent variables, when the dataset is
    switched

    :param input_dataset: The chosen dataset
    :return: The updates values for the dropdown of the independent variables
    """

    independent_variables = dataset_config[Dataset(input_dataset)][
        "training_data"
    ].columns
    return independent_variables, independent_variables[0]


# choose which x and y will be taken, depending on dataset.
# Maybe another dropdown/slider?
fig = px.scatter(df1, x=df1.columns[3], y=df1.columns[-2])

# dropdown for regression type
regression_type = dcc.Dropdown(
    options=["Linear Regression", "Locally Weighted Regression"],
    clearable=False,
    id="regression-dropdown",
    placeholder="Choose a regression type",
    value="Linear Regression",
)


# create a callback to combine both dropdowns with a scatter plot triggered by button to
# plot
@app.callback(
    Output(component_id="scatter-plot", component_property="figure"),
    Input(
        component_id="dataset-dropdown", component_property="value"
    ),  # input 1 -> selected_dataset
    Input(
        component_id="indep-dropdown-1", component_property="value"
    ),  # input 2 -> selected_indep_1
    Input(
        component_id="regression-dropdown", component_property="value"
    ),  # input 3 -> regression_type
)
def update_df(selected_dataset, selected_indep_1, regression_type):
    """
    Updates the regression models when the dataset, the choice of independent variable
    or when the type of regression model changes

    :param selected_dataset: The selcted dataset
    :param selected_indep_1: The selected independent variable
    :param regression_type: The selected regression type
    :return: The updated plot
    """

    dataset = dataset_config[Dataset(selected_dataset)]
    training_data = dataset["training_data"]
    dep = dataset["dependent_variable"]

    X = training_data[selected_indep_1].values
    y = dep.values

    current_figure = go.Figure()
    # linear regression fits
    if regression_type == "Linear Regression":
        # calculate regression
        lin_reg = LinearRegression([X, y], transposed=True, name="Linear Regression")
        x_range = np.linspace(min(X), max(X), 100)
        y_range = [lin_reg.predict(xi) for xi in x_range]

        # add to plot
        current_figure = go.Figure(
            [
                go.Scatter(x=X, y=y, mode="markers", name="Data"),  # dataplot
                go.Scatter(
                    x=x_range,  # regression line
                    y=y_range,
                    mode="lines",
                    name="Linear Regression Prediction",
                ),
            ]
        )
        current_figure.update_layout(title="Linear Regression")

    if regression_type == "Locally Weighted Regression":
        # calcaulate regression
        # tau = len(y)/2
        lwr = LocallyWeightedRegression(
            [X, y], transposed=True, name="Local Weighted Regression", tau=2
        )
        x_range = np.linspace(min(X), max(X), 50)
        y_range = [lwr.predict(xi) for xi in x_range]
        y_prime_range = localreg.localreg(X, y, x_range)

        # add to plot
        current_figure = go.Figure(
            [
                go.Scatter(x=X, y=y, mode="markers", name="Data"),
                go.Scatter(
                    x=x_range, y=y_prime_range, mode="lines", name="localreg Prediction"
                ),
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    name="Our LWR Prediction",
                    mode="lines",
                    line=dict(color="green"),
                ),
            ]
        )
        current_figure.update_layout(title="Local Weighted Regression ")

    return current_figure


mse_lin_reg = html.Div(
    [
        html.H4("Our Linear Regression MSE:"),
        html.Div([dcc.Textarea(id="mse-linreg", readOnly=True)]),
    ]
)

mae_lin_reg = html.Div(
    [
        html.H4("Our Linear Regression MAE:"),
        html.Div([dcc.Textarea(id="mae-linreg", readOnly=True)]),
    ]
)

mse_sk_lin_reg = html.Div(
    [
        html.H4("sklearn Linear Regression MSE:"),
        html.Div(
            [
                dcc.Textarea(
                    id="mse-sklinreg",
                    readOnly=True,
                )
            ]
        ),
    ]
)

mae_sk_lin_reg = html.Div(
    [
        html.H4("sklearn Linear Regression MAE:"),
        html.Div(
            [
                dcc.Textarea(
                    id="mae-sklinreg",
                    readOnly=True,
                )
            ]
        ),
    ]
)

mse_lwr = html.Div(
    [
        html.H4("Our LWR MSE:"),
        html.Div(
            [
                dcc.Textarea(
                    id="mse-lwr",
                    readOnly=True,
                )
            ]
        ),
    ]
)

mae_lwr = html.Div(
    [
        html.H4("Our LWR MAE:"),
        html.Div(
            [
                dcc.Textarea(
                    id="mae-lwr",
                    readOnly=True,
                )
            ]
        ),
    ]
)
mse_lclreg = html.Div(
    [
        html.H4("localreg LWR MSE:"),
        html.Div(
            [
                dcc.Textarea(
                    id="mse-localreg",
                    readOnly=True,
                )
            ]
        ),
    ]
)

mae_lclreg = html.Div(
    [
        html.H4("localreg LWR MAE:"),
        html.Div(
            [
                dcc.Textarea(
                    id="mae-localreg",
                    readOnly=True,
                )
            ]
        ),
    ]
)


@app.callback(
    Output("mse-linreg", "value"),
    Output("mae-linreg", "value"),
    Output("mse-sklinreg", "value"),
    Output("mae-sklinreg", "value"),
    Input("dataset-dropdown", "value"),
)
def update_linear_metrics(selected_dataset):
    """
    Updates the performance metrics of the locally weighted regression when the
    dataset changes

    :param selected_dataset: The selected dataset
    :return: The performance values of our implementation and the one of sklearn
    """

    if selected_dataset == "Prostate":
        df = df1  # choose the right dataframe
        X = df.iloc[:, 1:-2].to_numpy()
        y = df.iloc[:, -2].to_numpy()
        data = df.iloc[:, 1:-1].to_numpy()
    elif selected_dataset == "Wine Quality":
        df = df2
        X = df.iloc[:, 0:-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        data = df.iloc[:].to_numpy()
    elif selected_dataset == "Real Estate":
        df = df3
        X = df.iloc[:, 1:-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        data = df.iloc[:, 1:].to_numpy()

    # get MSE and MAE
    reg = LinearRegression(data, transposed=False, name="Linear Regression")
    mse_linreg = reg.get_mse()
    mae_linreg = reg.get_mae()

    skreg = linear_model.LinearRegression()
    skreg.fit(X, y)
    skreg_pred = skreg.predict(X)
    mse_sklinreg = metrics.mean_squared_error(y, skreg_pred)
    mae_sklinreg = metrics.mean_absolute_error(y, skreg_pred)
    return str(mse_linreg), str(mae_linreg), str(mse_sklinreg), str(mae_sklinreg)


@app.callback(
    Output("mse-lwr", "value"),
    Output("mae-lwr", "value"),
    Output("mse-localreg", "value"),
    Output("mae-localreg", "value"),
    Input("dataset-dropdown", "value"),
)
def update_lwr_metrics(selected_dataset):
    """
    Updates the performance metrics of the locally weighted regression when the
    dataset changes


    :param selected_dataset: The selected dataset
    :return: The performance values of our implementation and the one of local reg
    """

    if selected_dataset == "Prostate":
        df = df1  # choose the right dataframe
        X = df.iloc[:, 1:-2].to_numpy()
        y = df.iloc[:, -2].to_numpy()
        data = df.iloc[:, 1:-1].to_numpy()
    elif selected_dataset == "Wine Quality":
        df = df2
        X = df.iloc[:, 0:-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        data = df.iloc[:].to_numpy()
    elif selected_dataset == "Real Estate":
        df = df3
        X = df.iloc[:, 1:-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        data = df.iloc[:, 1:].to_numpy()

    # get MSE and MAE
    reg = LocallyWeightedRegression(
        data, transposed=False, name="Locally Weighted Regression", tau=30
    )
    mse_localwr = reg.get_mse()
    mae_localwr = reg.get_mae()

    y_pred = localreg.localreg(X, y)
    mse_localreg = metrics.mean_squared_error(y, y_pred)
    mae_localreg = metrics.mean_absolute_error(y, y_pred)
    return str(mse_localwr), str(mae_localwr), str(mse_localreg), str(mae_localreg)


# set the layout of the dashboard
app.layout = html.Div(
    children=[
        html.H2("Linear and Locally Weighted Regression on real data"),
        html.Br(),
        dataset_dropdown,
        html.Br(),
        indep_dropdown_1,
        html.Br(),
        regression_type,
        dcc.Graph(id="scatter-plot", figure=fig),
        html.Br(),
        html.Div(
            children=[
                html.H3(
                    "Regression metrics for selected dataset, regressed on ALL features"
                ),
                html.Div(mse_lin_reg, className="grid-item"),
                html.Div(mse_sk_lin_reg, className="grid-item"),
                html.Div(mae_lin_reg, className="grid-item"),
                html.Div(mae_sk_lin_reg, className="grid-item"),
                html.Div(mse_lwr, className="grid-item"),
                html.Div(mse_lclreg, className="grid-item"),
                html.Div(mae_lwr, className="grid-item"),
                html.Div(mae_lclreg, className="grid-item"),
            ],
            className="grid-container",
        ),
    ]
)


if __name__ == "__main__":
    app.run(debug=True, port=8500)
