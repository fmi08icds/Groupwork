import inspect
"""This is our interactive app to learn about """
# pylint: disable=W0603, W0718, W0123
import traceback

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import traceback
import dash_bootstrap_components as dbc
import dash_daq as daq
from regression_edu.data.simple_noise import generate_x, add_noise
from regression_edu.models.locally_weighted_regression import LocallyWeightedRegression
from regression_edu.models.linear_regression import LinearRegression
import math
import re

SECTIONS = None
NAME_LWR = "LWR"
NAME_LIN = "Linear Regression"
SIGMA = 1
TAU = 0.5


# external CSS stylesheets
external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css',
    {
        'href':"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
        'rel': 'stylesheet',
        'integrity': 'sha512-+7QzvzJ7yJvz3z9zJz8+JZx6v5zgjv9J5L5z3zJ9xKvZy5zJz7ZzZ6zvJZJ5zjLJ7L8yQzvJz1wzJzvzJzvzJw==',
        'crossorigin': 'anonymous',
        'referrerpolicy:': 'no-referrer',
    }
]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.icons.FONT_AWESOME,dbc.themes.BOOTSTRAP],
    )  # Bootstrap theme

SIDEBAR_WIDTH = 25

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": f"{SIDEBAR_WIDTH}rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": f"{SIDEBAR_WIDTH+2}rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

default = "2 * x + 2"
# dummy data

x,y = generate_x(lambda x: 2 * x + 2,size=10,loc=0,scale=1)
x,y = x, add_noise(y, distr_eps='normal',scale=1, size=10, loc=0)

reg_lwr = LocallyWeightedRegression(
    [x,y], transposed=True, name=NAME_LWR, sections=SECTIONS
)
reg_lin = LocallyWeightedRegression([x,y], transposed=True, name=NAME_LIN, tau=TAU, sigma=SIGMA)

data_generation_setting = dbc.Card(
    [
        dbc.CardHeader(html.H3("Data Setup")),
        dbc.CardBody(
            [
                html.H4("Function"),
                dcc.Input(
                    id="data_generation_function",
                    placeholder=default,
                ),
                html.H4("Sampling Mode for x"),
                dcc.Dropdown(id='x_distr',
                             # 
                             options=['Normal',
                                      'Uniform','Exponential'],
                             value="Normal"
                                    ),
                dbc.CardBody(id="sampling_x"),
                html.H4("# Samples"),
                dcc.Slider(
                    id="data_generation_samples",
                    className="slider",  ## for css reasons
                    min=0,
                    max=500,
                    step=1,
                    updatemode="drag",
                    value=10,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.H4("Range"),
                dcc.RangeSlider(
                    id="data_range",
                    className="slider",  ## for css reasons
                    min=-25,
                    max=25,
                    step=1,
                    value=[-3, 3],
                    allowCross=False,
                    dots=False,
                    updatemode="mouseup",
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.H4("Noise Level"),
                dcc.Slider(
                    id="noise_factor",
                    className="slider",  ## for css reasons
                    min=0,
                    max=1,
                    step=0.01,
                    updatemode="drag",
                    value=0.5,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.H4("Error Distribution"),
                dcc.Dropdown(id='error_distr',
                             # 
                             options=['Normal',
                                      'Uniform','Exponential','Poisson','Heteroscedastic'],
                             value="Normal"
                                    ),
                dbc.CardBody(id="sampling_error"),
                ### dynamically load distribution parameters ... going to be a little bit tricky
                html.Hr(),
                html.Div(
                    [dbc.Button("Regenerate Data", n_clicks=0, color="primary")],
                    className="d-grid gap-2",
                ),
            ]
        ),
    ]
)


@app.callback(
        [Output('sampling_x', 'children')],
        [Input('x_distr', 'value'),
         Input('data_generation_samples', 'value')]
)
def conditional_content_x(condition,sample_size):
    '''
    This function dynamically loads the conditional content for the error distribution
    from the docstring of the numpy random module.
    '''
    collect_x = getattr(np.random,str.lower(condition))
    doc_str = inspect.getdoc(collect_x)
    keystr=re.search(r'\((.*?)\)', string=doc_str)
    # Extract the parameter names and values
    matches = re.findall(r'(\w+)\s*=\s*([^,]+)', string=keystr.group(1))
    # Extract the parameter names
    parameter_names = parameter_names = [match[0] for match in matches]
    parameter_values = [eval(match[1]) for match in matches]

    content = []
    # dash-component factory:
    for param,val in zip(parameter_names,parameter_values):
        # create a dcc.Input for each parameter
        rpattern = rf'{param}\s*: [^.]*.'
        tooltip_text = re.search(pattern=rpattern,string=doc_str)[0]
        rpattern = r'\n([^.\n]*)'
        tooltip_text = re.search(pattern=rpattern,string=tooltip_text)[0]
        if param == 'size':
            content.append(dbc.InputGroup([dbc.InputGroupText(param,id="tooltip-target"+param,
                                                               style={"cursor": "pointer"}),
                                            dbc.Tooltip(tooltip_text, target="tooltip-target"+param),
                                            dcc.Input(id=param, placeholder=param, type='number',required=False,value=sample_size,disabled=True)])
                            )
        else:
            content.append(dbc.InputGroup([dbc.InputGroupText(param,id="tooltip-target"+param,
                                                               style={"cursor": "pointer"}),
                                            dbc.Tooltip(tooltip_text, target="tooltip-target"+param),
                                            dcc.Input(id=param, placeholder=param, type='number',required=False,value=val)])
            )



    return [content]


@app.callback(
        [Output('sampling_error', 'children')],
        [Input('error_distr', 'value'),
         Input('data_generation_samples', 'value')]
)
def conditional_content_error(condition,sample_size):
    '''
    This function dynamically loads the conditional content for the error distribution
    from the docstring of the numpy random module.
    '''
    print(condition)
    print(condition=='Heteroscedastic')

    if condition != 'Heteroscedastic':
        collect_x = getattr(np.random,str.lower(condition))
        doc_str = inspect.getdoc(collect_x)
        keystr=re.search(r'\((.*?)\)', string=doc_str)
        # Extract the parameter names and values
        matches = re.findall(r'(\w+)\s*=\s*([^,]+)', string=keystr.group(1))
        # Extract the parameter names
        parameter_names = parameter_names = [match[0] for match in matches]
        parameter_values = [eval(match[1]) for match in matches]

        content = []
        # dash-component factory:
        for param,val in zip(parameter_names,parameter_values):
            # create a dcc.Input for each parameter
            rpattern = rf'{param}\s*: [^.]*.'
            tooltip_text = re.search(pattern=rpattern,string=doc_str)[0]
            rpattern = r'\n([^.\n]*)'
            tooltip_text = re.search(pattern=rpattern,string=tooltip_text)[0]
            if param == 'size':
                content.append(dbc.InputGroup([dbc.InputGroupText(param,id="tooltip-target"+param,
                                                                style={"cursor": "pointer"}),
                                                dbc.Tooltip(tooltip_text, target="tooltip-target"+param),
                                                dcc.Input(id=param, placeholder=param, type='number',required=False,value=sample_size,disabled=True)])
                                )
            else:
                content.append(dbc.InputGroup([dbc.InputGroupText(param,id="tooltip-target"+param,
                                                                style={"cursor": "pointer"}),
                                                dbc.Tooltip(tooltip_text, target="tooltip-target"+param),
                                                dcc.Input(id=param, placeholder=param, type='number',required=False,value=val)])
                )



        return [content]
    else:
        # Radiobutton for direction of heteroscedasticity:
        content = []

        # Slider for intensity
        content.append(dcc.Slider(id='heteroscedasticity',
                                            min=-1,
                                            max=1,
                                            step=0.01,
                                            updatemode="drag",
                                            value=0,
                                            marks=None,
                                        ))                                         

        return [content]


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
    ]
)


model_input = dbc.Card(
    [dbc.CardHeader(html.H3("Model input")), dbc.CardBody([regression_equation])]
)


user_input = dbc.Row(
    [
        dbc.Col(model_input),
    ]
)

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
                        ),
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

tab_lr_content = dbc.Card(dbc.CardBody([html.P("This is tab 1")]), class_name="mt-3")

tab_lasso_content = dbc.Card(dbc.CardBody([html.P("This is tab 2")]), class_name="mt-3")

tab_lwr_content = dbc.Card(
    [
        dbc.CardHeader(html.H3("Parameters")),
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
    ],
    class_name="mt-3",
)

tabs = dbc.Tabs(
    [
        dbc.Tab(tab_lr_content, label="Linear Regression"),
        dbc.Tab(tab_lasso_content, label="Lasso Regression"),
        dbc.Tab(tab_lwr_content, label="Locally weighted Regression"),
    ]
)


sidebar = html.Div(
    [
        html.H2("Config", className="display-4"),
        html.Hr(),
        html.P(
            "Play around with the parameters and find out what changes",
            className="lead",
        ),
        tabs,
        dbc.Row(dbc.Col(data_generation_setting)),
    ],
    style=SIDEBAR_STYLE,
)


app.layout = dbc.Container(
    [
        html.H1("Group 3: Regression", className="text-center my-3"),
        html.H2("LWR and Linear Regression", className="text-center my-3"),
        html.Div(
            [
                dbc.Col(sidebar),
                dbc.Col(
                    [user_input, output],
                ),
            ]
        ),
    ]
)
#        dcc.Store(id="initial_data", data=generate_init_data(init_val.sigma_X1, init_val.sigma_X2, init_val.corr, init_val.mean_X1, init_val.mean_X2, MAX_SAMPLE_SIZE)),
#       dcc.Store(id="cur_data"),


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
        Input("data_range", "value"),
        Input("regression_equation_input", "value"),
        Input("sections", "value"),
        Input("tau", "value"),
        Input("sigma", "value"),
        Input("x_distr", "value"),
        Input("error_distr", "value"),
    ],
)
def update_regression(
    data_generation_function,
    data_generation_samples,
    noise_factor,
    data_range,
    regression_equation_input,
    sections,
    tau,
    sigma,
    x_distr,
    error_distr,
):
    """_summary_

    :param data_generation_function: _description_
    :param data_generation_samples: _description_
    :param noise_factor: _description_
    :param data_range: _description_
    :param regression_equation_input: _description_
    :param sections: _description_
    :param tau: _description_
    :param sigma: _description_
    :param error_distr: _description_
    :return: _description_
    """
    # set default for data generation function
    global reg_lwr
    global reg_lin
    data_generation_function = data_generation_function or default
    regression_equation_input = regression_equation_input or default

    x = y = None
    try:
        function = eval(f"lambda x:{data_generation_function}")
        x, y = generate_x(lambda x: 2 * x + 2,size=10,loc=0,scale=1)
        x, y = x,add_noise(y, distr_eps='normal',scale=1, size=10, loc=0)
        # update the data
        sections = None if sections is None or sections.strip() == "" else int(sections)
        tau = None if tau is None or tau.strip() == "" else float(tau)
        sigma = None if sigma is None or sigma.strip() == "" else float(sigma)
        reg_lwr = LocallyWeightedRegression(
            [x, y],
            transposed=True,
            name=NAME_LWR,
            sections=sections,
            tau=tau,
            sigma=sigma,
        )
        reg_lin = LinearRegression([x, y], transposed=True, name=NAME_LIN)
    except Exception:  # I will burn in hell for this
        print(traceback.print_exc())
        print("invalid function")

    # calculate the sum of squares
    sum_of_squares_lin = "{:,.2f}".format(reg_lin.get_sum_of_squares())
    sum_of_squares_lwr = "{:,.2f}".format(reg_lwr.get_sum_of_squares())

    # calculate the mean squared error
    mean_squared_error_lin = "{:,.2f}".format(reg_lin.get_mse())
    mean_squared_error_lwr = "{:,.2f}".format(reg_lwr.get_mse())

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
            y=[reg_lwr.predict(xi) for xi in x_lin],
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
