
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
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc


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


## header 
header = dbc.Col(html.H1('Regression: Prototype Teaching Tool', style={'textAlign':'center'}))


## mega unschön, refactor later.
sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout sliders", className="lead"
        ),
        dbc.Row([
            dbc.Col(html.Label('Sample Size', style={'textAlign':'left'}), width=3),
            dbc.Col(dcc.Slider(id='sample-size-slider',
                className='slider', ## for css reasons
                min=0,
                max=250,
                step=1,
                updatemode='drag',
                value=25,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ), width=9),
            dbc.Col(html.Label('sigma X1', style={'textAlign':'left'}), width=3),
            dbc.Col(dcc.Slider(id='sigma-X1-slider',
                className='slider', ## for css reasons
                min=0,
                max=10,
                step=0.1,
                updatemode='drag',
                value=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ), width=9),
            dbc.Col(html.Label('mean X1', style={'textAlign':'left'}), width=3),
            dbc.Col(dcc.Slider(id='mean-X1-slider',
                className='slider', ## for css reasons
                min=0,
                max=10,
                step=0.1,
                updatemode='drag',
                value=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ), width=9),
            dbc.Col(html.Label('mean X2', style={'textAlign':'left'}), width=3),
            dbc.Col(dcc.Slider(id='mean-X2-slider',
                className='slider', ## for css reasons
                min=0,
                max=10,
                step=0.1,
                updatemode='drag',
                value=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ), width=9),
            dbc.Col(html.Label('Sigma X2', style={'textAlign':'left'}), width=3),
            dbc.Col(dcc.Slider(id='sigma-X2-slider',
                className='slider', ## for css reasons
                min=0,
                max=10,
                step=0.1,
                updatemode='drag',
                value=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ), width=9),
            dbc.Col(html.Label('Correlation X1~X2', style={'textAlign':'left'}), width=3),
            dbc.Col(dcc.Slider(id='corr-X1-X2-slider',
                className='slider', ## for css reasons
                min=-1,
                max=1,
                step=0.1,
                updatemode='drag',
                value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ), width=9),
        ])
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE,
                   children=[
                                html.H2('Regression', style={'textAlign':'left'}),
                                dcc.Graph(figure={}, id='interactive-regression')
                    ]
)


app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
## create an app layout with a sidebar and a main content area





@callback(
    ## section 1: Data Generation
    Output('interactive-regression', 'figure'),

    # Synthesize the data
    ## controlling the distribution of X1 and X2
    Input('sample-size-slider', 'value'),
    Input('mean-X1-slider', 'value'),
    Input('mean-X2-slider', 'value'),
    Input('sigma-X1-slider', 'value'),
    Input('sigma-X2-slider', 'value'),
    Input('corr-X1-X2-slider', 'value'),
)

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

def update_graph(samplesize,
X1_mean,
X2_mean,
X1_sigma,
X2_sigma,
x1_x2_correlation):
    
    ## generate covariance matrix from inputs
    cov_mat = np.array([[X1_sigma**2, x1_x2_correlation*X1_sigma*X2_sigma], [x1_x2_correlation*X1_sigma*X2_sigma, X2_sigma**2]])

    mult_norm=np.random.multivariate_normal(mean=[X1_mean,X2_mean], cov=cov_mat, size=samplesize)
    
    ## trendline ... 
    ## regression call

    fig = px.scatter(x=mult_norm.T[0], y=mult_norm.T[1])
    return fig




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



if __name__ == '__main__':
    app.run_server(debug=True)



