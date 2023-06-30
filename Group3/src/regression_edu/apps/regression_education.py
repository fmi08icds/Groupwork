import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import traceback
import dash_bootstrap_components as dbc
import dash_daq as daq
from regression_edu.data.simple_uniform_noise import simple_uniform
from regression_edu.models.locally_weighted_regression import LocallyWeightedRegression
from regression_edu.models.linear_regression import LinearRegression

SECTIONS = None
NAME_LWR = "LWR"
NAME_LIN = "Linear Regression"
SIGMA = 1
TAU =.5


app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
)  # Bootstrap theme

default = "2 * x + 2"
# dummy data

data = simple_uniform(lambda x: 2 * x + 2, 100, (-100, 100), 0.5)
reg_lwr = LocallyWeightedRegression(
    data, transposed=True, name=NAME_LWR, sections=SECTIONS
)
reg_lin = LocallyWeightedRegression(data, transposed=True, name=NAME_LIN, tau=TAU, sigma=SIGMA)

data_generation_setting = dbc.Card(
    [
        dbc.CardHeader(html.H3("Data generation setting")),
        dbc.CardBody(
            [
                html.H4("Function"),
                dcc.Input(
                    id="data_generation_function",
                    placeholder=default,
                ),
                html.H4("Number of samples"),
                daq.NumericInput(
                    id="data_generation_samples", value=100, min=1, max=500
                ),
                html.H4("Noise variance"),
                daq.NumericInput(
                    id="noise_factor", value=0.5, min=0, max=1
                ),
                html.H4("Lower bound"),
                daq.NumericInput(
                    id="data_generation_lower", value=-100, min=-500, max=0
                ),
                html.H4("Upper bound"),
                daq.NumericInput(id="data_generation_upper", value=100, min=0, max=500),
            ]
        ),
    ]
)
regression_equation = dbc.Card(
    [
        dbc.CardHeader(html.H3("Regression equation")),
        dbc.CardBody(
            [
                html.H4("Function"),
                dcc.Input(
                    id="regression_equation_input",
                    placeholder=default,
                ),
            ]
        ),
        dbc.CardHeader(html.H3("Locally weighted regression")),
        dbc.CardBody(
            [
                html.H4("Sections"),
                dcc.Input(
                    id="sections",
                    placeholder=None,
                ),
                html.H4("Tau"),
                dcc.Input(
                    id="tau",
                    placeholder=TAU,
                ),
                html.H4("Sigma"),
                dcc.Input(
                    id="sigma",
                    placeholder=SIGMA,
                ),
            ]
        ),
    ]
)
model_input = dbc.Card(
    [dbc.CardHeader(html.H3("Model input")), dbc.CardBody([regression_equation])]
)


user_input = dbc.Row([dbc.Col(model_input), dbc.Col(data_generation_setting)])

equation_and_metrics = dbc.Card(
    [
        dbc.CardHeader(html.H3("Regression Equation and Metrics")),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Linear Regression"),
                                html.H5("Sum of Squares"),
                                html.P(id="sum_of_squares_lin"),
                                html.H5("Mean Squared Error"),
                                html.P(id="mean_squared_error_lin"),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H4("LW-Regression"),
                                html.H5("Sum of Squares"),
                                html.P(id="sum_of_squares_lwr"),
                                html.H5("Mean Squared Error"),
                                html.P(id="mean_squared_error_lwr"),
                            ]
                        )
                    ]
                ),
            ]
        ),
    ],
    className="my-3",
)
output = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                id="loading",
                type="circle",
                children=dcc.Graph(id="graph"),
            )
        ),
        dbc.Col(
            equation_and_metrics,
        ),
    ]
)

app.layout = dbc.Container(
    [
        html.H1("Group2: Regression", className="text-center my-3"),
        html.H2("LWR and Linear Regression", className="text-center my-3"),
        dbc.Col([user_input, output]),
    ]
)


@app.callback(
    [
        Output("graph", "figure"),
        Output("sum_of_squares_lin", "children"),
        Output("sum_of_squares_lwr", "children"),
        Output("mean_squared_error_lin", "children"),
        Output("mean_squared_error_lwr", "children"),
    ],
    [
        Input("data_generation_function", "value"),
        Input("data_generation_samples", "value"),
        Input("noise_factor", "value"),
        Input("data_generation_lower", "value"),
        Input("data_generation_upper", "value"),
        Input("regression_equation_input", "value"),
        Input("sections", "value"),
        Input("tau", "value"),
        Input("sigma", "value")
    ],
)
def update_regression(
    data_generation_function,
    data_generation_samples,
    noise_factor,
    data_generation_lower,
    data_generation_upper,
    regression_equation_input,
    sections,
    tau,
    sigma
):
    # set default for data generation function
    global reg_lwr
    global reg_lin
    data_generation_function = data_generation_function or default
    regression_equation_input = regression_equation_input or default

    x = y = None
    try:
        function = eval(f"lambda x:{data_generation_function}")
        x, y = simple_uniform(
            function,
            data_generation_samples,
            (data_generation_lower, data_generation_upper),
            noise_factor
        )
        # update the data
        sections = None if sections is None or sections.strip() == "" else int(sections)
        tau = None if tau is None or tau.strip() == "" else float(tau)
        sigma = None if sigma is None or sigma.strip() == "" else float(sigma)
        reg_lwr = LocallyWeightedRegression(
            [x, y], transposed=True, name=NAME_LWR, sections=sections, tau=tau, sigma=sigma
        )
        reg_lin = LinearRegression([x, y], transposed=True, name=NAME_LIN)
    except Exception:  # I will burn in hell for this
        print(traceback.print_exc())
        print("invalid function")

    # calculate the sum of squares
    sum_of_squares_lin = '{:,}'.format(reg_lin.get_sum_of_squares())
    sum_of_squares_lwr = '{:,}'.format(reg_lwr.get_sum_of_squares())

    # calculate the mean squared error
    mean_squared_error_lin = '{:,}'.format(reg_lin.get_MSE())
    mean_squared_error_lwr = '{:,}'.format(reg_lwr.get_MSE())

    # calculate the root mean squared error

    # create a figure with the data and regression line
    fig = go.Figure()

    # add a scatter trace for the data
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="blue"),
            name="Data",
        )
    )

    # add a scatter trace for the regression line
    fig.add_trace(
        go.Scatter(
            x=reg_lin.get_x_column(0),
            y=reg_lin.predicted_values,
            mode="lines",
            marker=dict(color="green"),
            name="Linear Regression Line",
        )
    )
    x_lin = np.linspace(min(reg_lwr.get_x_column(0)), max(reg_lwr.get_x_column(0)), 50)
    fig.add_trace(
        go.Scatter(
            x=x_lin,
            y=[reg_lwr.f(xi) for xi in x_lin],
            mode="lines",
            marker=dict(color="purple"),
            name="Locally Weighted Regression",
        )
    )

    # update the layout
    fig.update_layout(
        title="Interactive Regression",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        height=500,
    )

    # return the figure, error metrics, and regression equation
    return (
        fig,
        sum_of_squares_lin,
        sum_of_squares_lwr,
        mean_squared_error_lin,
        mean_squared_error_lwr,
    )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
