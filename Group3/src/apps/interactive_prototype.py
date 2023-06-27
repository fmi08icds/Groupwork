import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
from interactive_regression import InteractiveRegression
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
)  # Bootstrap theme

# dummy data
x_data = np.linspace(0, 10, 100)
y_data = 2 * x_data + 1 + np.random.randn(100) * 2

reg = InteractiveRegression(x_data, y_data, 0, 0, 0)

app.layout = dbc.Container(
    [
        html.H1("Interactive Regression", className="text-center my-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H3("Coefficients and Plot")),
                                dbc.CardBody(
                                    [
                                        html.H4("b0"),
                                        dcc.Slider(
                                            id="b0",
                                            min=-10,
                                            max=10,
                                            step=0.1,
                                            value=0,
                                            marks={i: str(i) for i in range(-10, 11)},
                                        ),
                                        html.Div(id="slider-output-b0"),
                                        html.H4("b1"),
                                        dcc.Slider(
                                            id="b1",
                                            min=-10,
                                            max=10,
                                            step=0.1,
                                            value=0,
                                            marks={i: str(i) for i in range(-10, 11)},
                                        ),
                                        html.Div(id="slider-output-b1"),
                                        html.H4("b2"),
                                        dcc.Slider(
                                            id="b2",
                                            min=-10,
                                            max=10,
                                            step=0.1,
                                            value=0,
                                            marks={i: str(i) for i in range(-10, 11)},
                                        ),
                                        html.Div(id="slider-output-b2"),
                                        dcc.Loading(
                                            id="loading",
                                            type="circle",
                                            children=dcc.Graph(id="graph"),
                                        ),
                                    ]
                                ),
                            ],
                            className="my-3",
                        ),
                    ],
                    width=8,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H3("Regression Equation and Metrics")
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5("Regression Equation"),
                                        html.P(id="regression_equation"),
                                        html.H5("Sum of Squares"),
                                        html.P(id="sum_of_squares"),
                                        html.H5("Mean Squared Error"),
                                        html.P(id="mean_squared_error"),
                                        html.H5("Root Mean Squared Error"),
                                        html.P(id="root_mean_squared_error"),
                                    ]
                                ),
                            ],
                            className="my-3",
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
    ]
)


@app.callback(
    dash.dependencies.Output("slider-output-b0", "children"),
    [dash.dependencies.Input("b0", "value")],
)
def update_output_b0(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output("slider-output-b1", "children"),
    [dash.dependencies.Input("b1", "value")],
)
def update_output_b1(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output("slider-output-b2", "children"),
    [dash.dependencies.Input("b2", "value")],
)
def update_output_b2(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    [
        Output("graph", "figure"),
        Output("sum_of_squares", "children"),
        Output("mean_squared_error", "children"),
        Output("root_mean_squared_error", "children"),
        Output("regression_equation", "children"),
    ],
    [Input("b0", "value"), Input("b1", "value"), Input("b2", "value")],
)

def update_regression(b0, b1, b2):
    # update the coefficients
    reg.b0 = b0
    reg.b1 = b1
    reg.b2 = b2

    # calculate the predicted values
    reg.calc_predicted_values()

    # calculate the residuals
    reg.calc_residuals()

    # calculate the sum of squares
    sum_of_squares = reg.calc_sum_of_squares()

    # calculate the mean squared error
    mean_squared_error = reg.calc_mean_squared_error()

    # calculate the root mean squared error
    root_mean_squared_error = reg.calc_root_mean_squared_error()

    # create a figure with the data and regression line
    fig = go.Figure()

    # add a scatter trace for the data
    fig.add_trace(
        go.Scatter(
            x=x_data, y=y_data, mode="markers", marker=dict(color="blue"), name="Data"
        )
    )

    # add a scatter trace for the regression line
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=reg.predicted_values,
            mode="lines",
            marker=dict(color="red"),
            name="Regression Line",
        )
    )

    # update the layout
    fig.update_layout(
        title="Interactive Regression",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True,
    )

    # format the regression equation
    equation = f"y = {b0} + {b1}x + {b2}x^2"

    # return the figure, error metrics, and regression equation
    return fig, sum_of_squares, mean_squared_error, root_mean_squared_error, equation


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
