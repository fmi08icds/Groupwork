import inspect
"""This is our interactive app to learn about """
# pylint: disable=W0603, W0718, W0123
import traceback

import dash
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import traceback
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.exceptions import PreventUpdate
from regression_edu.data.simple_noise import generate_x, add_noise
from regression_edu.models.locally_weighted_regression import LocallyWeightedRegression
from regression_edu.models.linear_regression import LinearRegression
import math
import re
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

SECTIONS = None
NAME_LWR = "LWR"
NAME_LIN = "Linear Regression"
SIGMA = 1
TAU = 0.5
MAX_SAMPLE_SIZE = 100

COLS = px.colors.qualitative.T10

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



SIDEBAR_WIDTH = 30

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": f"{SIDEBAR_WIDTH}rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

TOP_BAR_STYLE = {
  #'position': 'sticky',
  "top": '0px',
  'right': '20px',
  "width": '100%',
  "overflow": 'hidden',
#  'border': '3px solid #73AD21'
}

CONTENT_STYLE = {
    "margin-left": f"{SIDEBAR_WIDTH+2}rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

default = "5 * x + 10"

data_generation_setting = dbc.Card(
    [
        dbc.CardHeader(html.H3("Data Setup")),
        dbc.CardBody(
            [
                html.H4("Function"),
                dcc.Input(
                    id="data_generation_function",
                    placeholder=default,
                    debounce=True,
                    style={"cursor": "pointer"},
                ),
                dbc.Tooltip("a + b * x", target="data_generation_function"),
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
                    max=200,
                    step=1,
                    updatemode="drag",
                    value=25,
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
                html.Hr(),
                html.Div(
                    [dbc.Button("Regenerate Data", id="Regenerate Data", n_clicks=0, color="primary")],
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
         Input('data_generation_samples', 'value')],
)

def conditional_content_error(condition,sample_size):
    '''
    This function dynamically loads the conditional content for the error distribution
    from the docstring of the numpy random module.
    '''
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
        content = []
        #intensity
        param = 'heteroscedacity'
        content.append(dbc.InputGroup([dbc.InputGroupText(param,id="tooltip-target"+param,
                                                                style={"cursor": "pointer"}),
                                                dbc.Tooltip('intensity of heteroscedacity effect', target="tooltip-target"+param),
                                                dcc.Input(id=param, placeholder=param, type='number',required=False,value=0.5,disabled=False)]))
        
        param = 'size'
        content.append(dbc.InputGroup([dbc.InputGroupText(param,id="tooltip-target"+param,
                                                               style={"cursor": "pointer"}),
                                            dbc.Tooltip('Number of Observations', target="tooltip-target"+param),
                                            dcc.Input(id=param, placeholder=param, type='number',required=False,value=sample_size,disabled=True)]))
        return [content]


regression_equation = dbc.Card(
    [
        dbc.CardHeader(html.H3("Manuell Regression")),
        dbc.CardBody(
            [
                html.H4("Function"),
                dcc.Input(
                    id="regression_equation_input",
                    placeholder=default,
                    type="text",
                    value=default,

                ),
                dbc.Col([
                    dbc.Row(
                            [html.H5("Intercept"),
                                dcc.Slider(
                                    id="beta0",
                                    min=-25,
                                    max=25,
                                    step=0.1,
                                    updatemode="mouseup",
                                    value=0,
                                    marks=None,
                                    tooltip={"placement": "right", "always_visible": True},
                                ),
                                html.H5("Beta"),
                                dcc.Slider(
                                    id="beta1",
                                    min=-25,
                                    max=25,
                                    step=0.1,
                                    updatemode="mouseup",
                                    value=0,
                                    marks=None,
                                    tooltip={"placement": "right", "always_visible": True},
                                    )]
                        ),]
                )])
                    ]
)


model_input = dbc.CardBody([regression_equation])

data_output = html.Div(
    dbc.Row(
        [dbc.Col([dbc.CardHeader(html.H3('Data')),
                 dash_table.DataTable(data=[{}], id="table")]),
        dbc.Col([
        dbc.CardHeader(html.H3('Prediction')),
                 dash_table.DataTable(data=[{}], id="pred_table")])
                 ])
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
                                html.H5("Coefficients"),
                                html.H6('Intercept:'),html.P(id="lr_coefficients_int"),
                                html.H6('Beta:'),html.P(id="lr_coefficients_beta"),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H4("LW-Regression"),
                                html.H5("Sum of Squares"),
                                html.P(id="sum_of_squares_lwr"),
                                html.H5("Mean Squared Error"),
                                html.P(id="mean_squared_error_lwr"),
                        dbc.Row(
                            [
                                html.H4("Manual-Regression"),
                                html.H5("Sum of Squares"),
                                html.P(id="sum_of_squares_man_lin"),
                                html.H5("Mean Squared Error"),
                                html.P(id="mean_sq_error_man_lin"),
                            ])]
                        ),
                    ]
                ),
            ]
        ),
    ],
    className="my-3",
)

user_input = dbc.Row(
    [
        dbc.Col(equation_and_metrics),
        dbc.Row(data_output),
    ]
)

output = dbc.Col(
    [
        dbc.Row(
            [
                dcc.Loading(
                id="loading",
                type="circle",
                children=dcc.Graph(id="graph"),
            ),
            html.H4(id="Residuals"),
            dcc.Dropdown(id='residual_radio',
                        options=[
       {'label': 'Linear Regression', 'value': 0},
       {'label': 'Manuell Regression', 'value': 1},
       {'label': 'Locally Weighted Regression', 'value': 2}],
       multi=True,
       value=[0,2],
       ),
            dcc.Loading(
                id="loading-2",
                type="circle",
                children=dcc.Graph(id="eps_graph"),
            )]
        )
    ]
)

tab_lr_content = model_input

tab_lasso_content = dbc.Card(dbc.CardBody([html.P("This is tab 2")]), class_name="mt-3")

tab_lwr_content = dbc.Card(
    [
        dbc.CardHeader(html.H3("Parameters")),
        dbc.CardBody(
            [
                html.H4(
                    [
                        html.Span("Sections", id="sections_header")
                    ]
                ),
                dbc.Tooltip(
                    "The sections variable controls the number of local regressions that should be created. The default is that there is a section for each datapoint."
                    , target="sections_header"
                ),
                dcc.Input(
                    id="sections",
                    placeholder=10,
                    debounce=False,
                    type='number'
                ),
                html.H4(
                    [
                        html.Span("Tau", id="tau_header")
                    ]
                ),
                dbc.Tooltip(
                    "The tau value is responsible for how much the surrounding data points get considered when calculating x."
                    " The higher the value the higher the weights of sections."
                    , target="tau_header"
                ),
                dcc.Input(
                    id="tau",
                    placeholder=TAU,
                    debounce=False,
                    type='number'
                ),
                html.H4(
                    [
                        html.Span("Sigma", id="sigma_header")
                    ]
                ),
                dbc.Tooltip(
                    "Sigma describes the variance of the gaussian distribution and by default it is 1. The larger the value the smoother is the function."
                    , target="sigma_header"
                ),
                dcc.Input(
                    id="sigma",
                    placeholder=SIGMA,
                    debounce=False,
                    type='number'
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
        dbc.Tab(tab_lwr_content, label="Locally weighted Regression", id="LWR"),
        dbc.Tooltip(
            """
                Hyperparameters:\n
                 \t- sections, tau, sigma\n
                Functionality:\n
                 \t- The area gets devided in sections. By default, the number of sections equals the number of samples.\n
                 \t- For each section, a linearly weighted regression gets composed. This is a linear regression in which each datapoint is weighted according to its importance.\n
                 \t- The weight function for a section is w(i) = e^(-||centre - x_i||_2^2 / (2*tau^2).\n
                 \t- the centre is determined by sorting the sectioned data along the first factor and taking the median.\n 
                 \t- The independent local regressions get combined to one continuous regression by applying the normalised gaussian function as a smoothening function.\n
                 \t- The gaussian function is gauss(centre, x) = e^(-(centre-x)^2/(2*sigma^2)\n
                 \t- In order to prevent that values further away from the clusters going to zero, the values gets normalised by having the weights of the sum of gaussian functions sum up 1.\n
                 \t- Therefore, when predicting the value x, it is calculated by iterating through the local regressions and weighting them through gauss(centre, x)/sum(gaussian functions at x)\n
                 """
            , target="LWR"
        ),
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

init_data = np.transpose(generate_x(f=eval(f"lambda x:{default}"),distr_x='normal',loc=10,scale=5,size=MAX_SAMPLE_SIZE))





app.layout = dbc.Container(
    [
        dcc.Store(id="initial_data", data=init_data),
        dcc.Store(id="cur_data"),
        dcc.Store(id='pred_data'),
        dcc.Store(id='pred_data_lwr'),
        html.H1("Group 3: Regression", className="text-center my-3",style=TOP_BAR_STYLE),
        html.H2("LWR and Linear Regression", className="text-center my-3"),
        html.Div(
            [
                dbc.Col(sidebar),
                dbc.Col(
                    [output,user_input, ],
                ),
            ]
        ),
    ]
)



@app.callback(
    [
        Output("graph", "figure"),
        Output("sum_of_squares_lin", "children"),
        Output("sum_of_squares_lwr", "children"),
        Output("mean_squared_error_lin", "children"),
        Output("mean_squared_error_lwr", "children"),
        Output('lr_coefficients_int','children'),
        Output('lr_coefficients_beta','children'),

    ],
    [
        Input("sections", "value"),
        Input("tau", "value"),
        Input("sigma", "value"),
        Input('cur_data','data')
    ],prevent_initial_call=True
)
def update_regression(
    sections,
    tau,
    sigma,
    cur_data
):
    """_summary_

    :param data_generation_function: _description_
    :param data_generation_samples: _description_
    :param noise_factor: _description_
    :param data_range: _description_
    :param sections: _description_
    :param tau: _description_
    :param sigma: _description_
    :param error_distr: _description_
    :return: _description_
    """
    # set default for data generation function
    global reg_lwr
    global reg_lin
    
    x,y = np.transpose(cur_data)

    reg_lwr = LocallyWeightedRegression(
        [x, y],
        transposed=True,
        name=NAME_LWR,
        sections=sections,
        tau=tau,
        sigma=sigma,
    )
    reg_lin = LinearRegression([x, y], transposed=True, name=NAME_LIN)
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
            marker=dict(color=COLS[0]),
            name="Data",
        )
    )

    fig.add_trace(go.Scatter(
            mode="lines",
            marker=dict(color=COLS[2]),
            name="Manual Regression Line",
        ))

    # add a scatter trace for the regression line
    fig.add_trace(
        go.Scatter(
            x=reg_lin.get_x_column(0),
            y=reg_lin.predicted_values,
            mode="lines",
            marker=dict(color=COLS[1]),
            name="Linear Regression Line",
        )
    )
    x_lin = np.linspace(min(reg_lwr.get_x_column(0)), max(reg_lwr.get_x_column(0)), 50)
    fig.add_trace(
        go.Scatter(
            x=x_lin,
            y=[reg_lwr.predict(xi) for xi in x_lin],
            mode="lines",
            marker=dict(color=COLS[3]),
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
        round(reg_lin.coefficients[0],3),
        round(reg_lin.coefficients[1],3)
    )


@app.callback(
   [Output("cur_data", "data")],
   [Input("initial_data", "data"),
    Input("data_generation_function", "value"),
    Input("x_distr", "value"),
    Input("error_distr", "value"),
    Input('Regenerate Data', 'n_clicks'),
    State('sampling_x', 'children'),
    State('sampling_error', 'children'),
    ],prevent_initial_call=True)
def update_data(current_data, data_generation_function, x_distr,error_distr,button,sampling_x,sampling_error):
    x = y = None
    data_generation_function = data_generation_function or default
    print(f'function: {data_generation_function}')
    try:
        f = eval(f"lambda x:{data_generation_function}")

        ids, values = [],[]
        for prop in sampling_x:
            ids.append(prop['props']['children'][2]['props']['id'])
            values.append(prop['props']['children'][2]['props']['value'])

        err_ids, err_values = [],[]
        for prop in sampling_error:
            err_ids.append(prop['props']['children'][2]['props']['id'])
            err_values.append(prop['props']['children'][2]['props']['value'])


        x, y = generate_x(f, distr_x=x_distr, **dict(zip(ids, values)))
        y = add_noise(y, distr_eps=error_distr, **dict(zip(err_ids, err_values)))

        current_data= np.transpose([x,y])
    except Exception:
        print(traceback.print_exc())
        print("invalid function")
    return [current_data]

 

@app.callback(
    [Output('pred_data', 'data')],
    [Input('cur_data', 'data'),
     Input('beta0', 'value'),
     Input('beta1', 'value'),]
    )
def update_predicted_y_in_data(data,beta0,beta1):
    if data:
        x,y = np.transpose(data)
        lin_reg = LinearRegression([x,y], transposed=True, name=NAME_LIN)
        man_prediction = beta0 + beta1*x
        cur_data = np.transpose([lin_reg.predicted_values,man_prediction])
        return [cur_data]
    else:
        raise PreventUpdate

@app.callback([Output('pred_data_lwr', 'data')],
              [Input('cur_data', 'data'),
               Input("sections", "value"),
               Input("tau", "value"),
               Input("sigma", "value")],
               prevent_initial_call=True)
def update_predicted_lwr_y_in_data(data,sections,tau,sigma):
    if ctx.triggered_id == 'cur_data'\
    or ctx.triggered_id == 'sections'\
    or ctx.triggered_id == 'tau'\
    or ctx.triggered_id == 'sigma':
        x,y = np.transpose(data)
        reg_lwr = LocallyWeightedRegression([x,y],transposed=True,name=NAME_LWR,sections=sections,tau=tau,sigma=sigma)
        pred_lwr = [reg_lwr.predict(xi) for xi in x]
        return [pred_lwr]
    else:
        raise PreventUpdate


@app.callback([Output('table', 'data')],
              [Input('data_generation_samples', 'value'),
               Input('cur_data', 'data')],
               )
def update_table(N,data):
    out = pd.DataFrame(data, columns=['x','y'])
    out = round(out, 3)
    return [out.iloc[:N,:].to_dict('records')]
    
@app.callback([Output('pred_table', 'data')],
              [Input('pred_data', 'data'),
               Input('pred_data_lwr', 'data'),
               Input('data_generation_samples', 'value')],
               prevent_initial_call=False)
def update_pred_table(pred, pred_lwr,N):
    if ctx.triggered_id == 'pred_data'\
    or ctx.triggered_id == 'pred_data_lwr'\
    or ctx.triggered_id == 'data_generation_samples':
        col1, col2 = np.transpose(pred)
        col3 = np.transpose(pred_lwr)
        out = pd.DataFrame(data=np.transpose([col1,col2,col3]), columns=['y_LinReg','y_manReg', 'y_LWReg'])
        out = round(out, 3)
        return [out.iloc[:N,:].to_dict('records')]
    else:
        raise PreventUpdate


@app.callback([Output('regression_equation_input','value')],
               [Input('beta0', 'value'),
               Input('beta1','value')],)
def update_regression_equation(beta0,beta1):
    return [f'y = {beta0} + {beta1}*x']


@app.callback([Output('graph','figure',allow_duplicate=True),
               Output('sum_of_squares_man_lin', 'children'),
               Output('mean_sq_error_man_lin', 'children')],
              [Input('regression_equation_input', 'value'),
               State('graph','figure'),
               State('cur_data', 'data'),
               State('beta0', 'value'),
               State('beta1','value')],
              prevent_initial_call=True)
def build_custom_line(coeffs, figure:go.Figure, data:list, beta0, beta1):
    #regression_equation_input = regression_equation_input or default
    if ctx.triggered_id == "regression_equation_input":
        fig = go.Figure(figure)
        if data:
            x,y = np.transpose(data)
            # add a scatter trace for the data    
            predicted_values = (lambda x: beta0 + beta1*x)(x)
        
            sum_of_squares_lwr = np.sum((predicted_values - y) ** 2)
            mean_sq_error_man_lin = sum_of_squares_lwr / len(y)
            
            sum_of_squares_lwr = "{:,.2f}".format(sum_of_squares_lwr)
            mean_sq_error_man_lin = "{:,.2f}".format(mean_sq_error_man_lin)

            # add a scatter trace for the regression line
            fig.update_traces(x=x,
                            y=predicted_values,
                            name="Manual Regression Line",
                            selector=dict(name="Manual Regression Line"),
                            overwrite=True)
        
            return [fig,sum_of_squares_lwr,mean_sq_error_man_lin]
        else:
            raise PreventUpdate


@app.callback([Output('eps_graph','figure')],
                [State('cur_data', 'data'),
                 Input('residual_radio', 'value'),
                 Input('pred_table', 'data')],
                prevent_initial_call=True)
def build_eps_graph(data,col_id,predictions):
    if ctx.triggered_id == "pred_table" or ctx.triggered_id == "residual_radio":
        data = pd.DataFrame(data)
        pred = pd.DataFrame(predictions)
        traces=[]
        for id in col_id:
            traces.append(go.Scatter(name=pred.columns[id],
                x=data[1], y=data[1]-pred.iloc[:,id],
                mode='markers', marker=dict(color=COLS[id+1])))
            traces.append(go.Histogram(name=pred.columns[id],
                x=data[1]-pred.iloc[:,id],
                histnorm='probability density',xaxis="x2",
                yaxis="y2", marker=dict(color=COLS[id+1])))
        
        layout = go.Layout(
                xaxis=dict(
                    domain=[0, 0.6]
                ),
                xaxis2=dict(
                    domain=[0.65, 1]
                ),
                yaxis2=dict(
                    anchor="x2"
                )
            )    


        fig = go.Figure(data=traces, layout=layout)        
        fig.update_layout(
            title="Residuals vs. Predicted Values",
            xaxis_title="y",
            yaxis_title="Residuals",
            template="plotly_white",
            height=500,
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1 # gap between bars of the same location coordinates
            )
        return [fig]
    else:
        raise PreventUpdate
                 

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
