from dash import Dash, html, dcc, Output, Input, State
from dash.exceptions import PreventUpdate
import numpy as np


## Minimal example on how to use dcc.Store and manipulate its data. 
## there might be a way to do this with just one store but this 
## is way less confusing.

## approach:
## 1. create a static/constant dcc.Store 
## 2. create a dcc.Store that has the current transformation (can be basically any widget, store is just a abstraction)
## 3. add upon the constant store data the current transformation store data

app = Dash(__name__)


## layout has just 2 visible elements, a slider and a div.
## 2 stores are created, one for the initial value and one for the transformation.

app.layout = html.Div([
    dcc.Store(id='transformation', data=0),
    dcc.Store(id='init-memory', data=15),
    html.Div(id='transformation-output'),
    dcc.Slider(
        id='add-value',
        min=0,
        max=10,
        step=1,
        value=0,
    ),])

## watch changes to input value, one is just static and in the bg but is needed
## as an argument to give it to output value.
## function takes as argument current input value and transforms it
## and returns it as output value.
## meaning we do not ever operate on the init Store. 
@app.callback(Output('transformation', 'data'),
              Input('init-memory', 'data'),
              Input('add-value', 'value'))
def set_store_value(init, value):
    print('state', init)
    print('value', value)
    return init + value

## shows the transformed init-memory value in the transformation-output div.
@app.callback(Output('transformation-output', 'children'),
              Input('transformation', 'data'))
def get_output(value):
    return value

if __name__ == '__main__':
    app.run_server(debug=True, port=8077, threaded=True)