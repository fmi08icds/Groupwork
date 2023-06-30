## in this prototype i want to show and suggest
## building a little interactive teaching tool
## that shows the different aspects of regression
## given a synthetic dataset that can be
## adjusted by the user

# Define the synthetic dataset with adjustable parameters.
# Implement a regression model that can be trained on the dataset.
# Create an interactive interface that allows the user to adjust the dataset parameters and see the resulting regression model.
# Display the different aspects of regression, such as the regression line, residuals, and R-squared value.


import numpy as np
from numpy import random
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input, ctx
import dash_bootstrap_components as dbc
from dash import dash_table
from dataclasses import dataclass
from dash.exceptions import PreventUpdate
## constants

MAX_SAMPLE_SIZE = 250

## init widgets with initial data
@dataclass
class InitVal:
    sample_size= 25
    sigma_X1= 1
    mean_X1= 0
    sigma_X2= 1
    mean_X2= 0
    corr=0

init_val = InitVal()

app = Dash(external_stylesheets=[dbc.themes.SPACELAB])

# the style arguments for the sidebar. We use position:fixed and a fixed width

sidebar_width = 25

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": f"{sidebar_width}rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": f"{sidebar_width+2}rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


## create synthetic data via numpy
"""
the idea is not to sample from a distribution but to create a
simple linear model with some noise drawn from a certain distribution.
therefore we just need 2 points to draw a line and then add some noise.
toDo: outsource to own file. 
future features:
                1. not only have a linear model but also any polynomial.
                [checkbox for term of degree n up to... maybe 5?]
                2. add other distributions for noise.
                [dropdown menu?]
                3. heteroscedasticity and homoscedasticity.
                [can be added as checkbox buttons?]

"""


## construct y.

## by drawing X1,X2 from multivariate normal distribution
def generate_init_data(sigma_X1, sigma_X2, x1_x2_correlation, mean_X1, mean_X2, size):
    ## per default generate 250 (max) samples, so we never have to redraw because of sample_size changes
    ## just redraw if necessary, and that is only if, e.g., the correlation term changes.
    cov_mat = np.array(
            [
                [sigma_X1**2, x1_x2_correlation * sigma_X1 * sigma_X2],
                [x1_x2_correlation * sigma_X1 * sigma_X2, sigma_X2**2],
            ]
        )
    data = np.random.multivariate_normal(
                mean=[mean_X1, mean_X2], cov=cov_mat, size=size)
    return data 



## header
header = dbc.Col(
    html.H1("Regression: Prototype Teaching Tool", style={"textAlign": "center"})
)


## mega unschön, refactor later.
sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P("A simple sidebar layout sliders", className="lead"),
        dbc.Row(
            [
                dbc.Col(
                    html.Label("Sample Size", style={"textAlign": "left"}), width=3
                ),
                dbc.Col(
                    dcc.Slider(
                        id="sample-size-slider",
                        className="slider",  ## for css reasons
                        min=0,
                        max=MAX_SAMPLE_SIZE,
                        step=1,
                        updatemode="drag",
                        value=init_val.sample_size,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    width=9,
                ),
                dbc.Col(html.Label("sigma X1", style={"textAlign": "left"}), width=3),
                dbc.Col(
                    dcc.Slider(
                        id="sigma-X1-slider",
                        className="slider",  ## for css reasons
                        min=0,
                        max=10,
                        step=0.1,
                        updatemode="drag",
                        value=init_val.sigma_X1,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    width=9,
                ),
                dbc.Col(html.Label("mean X1", style={"textAlign": "left"}), width=3),
                dbc.Col(
                    dcc.Slider(
                        id="mean-X1-slider",
                        className="slider",  ## for css reasons
                        min=0,
                        max=10,
                        step=0.1,
                        updatemode="drag",
                        value=init_val.mean_X2,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    width=9,
                ),
                dbc.Col(html.Label("mean X2", style={"textAlign": "left"}), width=3),
                dbc.Col(
                    dcc.Slider(
                        id="mean-X2-slider",
                        className="slider",  ## for css reasons
                        min=0,
                        max=10,
                        step=0.1,
                        updatemode="drag",
                        value=init_val.mean_X2,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    width=9,
                ),
                dbc.Col(html.Label("Sigma X2", style={"textAlign": "left"}), width=3),
                dbc.Col(
                    dcc.Slider(
                        id="sigma-X2-slider",
                        className="slider",  ## for css reasons
                        min=0,
                        max=10,
                        step=0.1,
                        updatemode="drag",
                        value=init_val.sigma_X2,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    width=9,
                ),
                dbc.Col(
                    html.Label("Correlation X1~X2", style={"textAlign": "left"}),
                    width=3,
                ),
                dbc.Col(
                    dcc.Slider(
                        id="corr-X1-X2-slider",
                        className="slider",  ## for css reasons
                        min=-1,
                        max=1,
                        step=0.1,
                        updatemode="drag",
                        value=init_val.corr,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    width=9,
                ),
            ]
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="page-content",
    style=CONTENT_STYLE,
    children=[
        html.H2("Regression", style={"textAlign": "left"}),
        dcc.Graph(figure={}, id="interactive-regression"),
        dash_table.DataTable(data=[{}], id="table"),
    ],
)


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        content,
        dcc.Store(id="initial_data", data=generate_init_data(init_val.sigma_X1, init_val.sigma_X2, init_val.corr, init_val.mean_X1, init_val.mean_X2, MAX_SAMPLE_SIZE)),
        dcc.Store(id="cur_data"),
    ]
)
## create an app layout with a sidebar and a main content area


#    Input('x1-x2-correlation-type', 'value'),
#    Input('x1-distribution', 'value'),
#    Input('x2-distribution', 'value'),
#    Input('noise-distribution', 'value'),
#    Input('noise-correlation-slider', 'value'),
#    Input('noise-correlation-type', 'value'),
#    Input('noise-sigma-slider', 'value'),
#    Input('noise-mean-slider', 'value')

## multivariate gaussian distribution
## https://en.wikipedia.org/wiki/Multivariate_normal_distribution


## Jonny: okay - die lageparameter machen übertrieben keinen sinn. die daten sehen immer gleich aus,
## egal wo sie liegen. wobei .. wäre interessant zu sehen, wie sich MSE usw. verhält, wenn man
## die daten skaliert oder nicht, weil dann der wertebereich zw. [0,1] und sonst ja nicht ist.
## --> radiobuttons :P ...


## auserdem radiobutton für "eigene Linie"
## oder "Linie aus Daten berechnen"
## und das dann für alle verfahren, also basic ols
## ElasticNet und local linear regression.


@app.callback(
    Output("cur_data", "data"),
    Input("initial_data", "data"), ## dont ever touch initial data.
    
    ## section 1: Data Generation
    # Synthesize the data
    ## controlling the distribution of X1 and X2
    Input("sample-size-slider", "value"),
    Input("mean-X1-slider", "value"),
    Input("mean-X2-slider", "value"),
    Input("sigma-X1-slider", "value"),
    Input("sigma-X2-slider", "value"),
)
def update_data(current_data,
                sample_size,
                X1_mean, X2_mean,
                X1_sigma, X2_sigma):
    if ctx.triggered_id == None or ctx.triggered_id != "initial_data.data":
        current_data = np.array(current_data[:sample_size])
        sigma = np.array([X1_sigma, X2_sigma])
        updated_data = current_data * sigma + np.array([X1_mean, X2_mean])
        return updated_data

@app.callback(Output('table', 'data'),Input('cur_data','data'))
def update_table(data):
    return pd.DataFrame(data, columns=['x1','x2']).to_dict('records')


@app.callback(Output('interactive-regression', 'figure'),
              Input('cur_data', 'data'))
def update_scatter(data):
    data = pd.DataFrame(data, columns=['x1','x2'])
    
    fig = px.scatter(x=data['x1'],y=data['x2'])
    return fig



@app.callback(Output('initial_data', 'data'),
              Input('sigma-X1-slider', 'value'),
              Input('sigma-X2-slider', 'value'),
              Input('corr-X1-X2-slider', 'value'),
              Input('mean-X1-slider', 'value'),
              Input('mean-X2-slider', 'value'),
              prevent_initial_call=True)
def reinit_data(sigma_X1, sigma_X2, x1_x2_correlation, mean_X1, mean_X2):
    
    ## per default generate 250 (max) samples, so we never have to redraw because of sample_size changes
    ## just redraw if necessary, and that is only if, e.g., the correlation term changes.
    print(ctx.triggered_id)
    if ctx.triggered_id == 'corr-X1-X2-slider':
        print('triggered')
        cov_mat = np.array(
                [
                    [sigma_X1**2, x1_x2_correlation * sigma_X1 * sigma_X2],
                    [x1_x2_correlation * sigma_X1 * sigma_X2, sigma_X2**2],
                ]
            )
        data = np.random.multivariate_normal(
                    mean=[mean_X1, mean_X2], cov=cov_mat, size=MAX_SAMPLE_SIZE)
        return data
    else:
        raise PreventUpdate


## das mit dem callback ... wir haben ja nur eine url, das pathing kann noch weg.
## habe das irgendwo her kopiert :D und wills gerade nicht kaputt machen.
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return content
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run_server(debug=True)
