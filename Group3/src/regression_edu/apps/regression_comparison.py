from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
from regression_edu.models.linear_regression import LinearRegression
from regression_edu.models.locally_weighted_regression import LocallyWeightedRegression
import localreg

#LocallyWeightedRegression()

# initialise dash app
app = Dash(__name__)

#  create datasets
df1 = pd.read_excel(io = '../data/realworld/prostate.xlsx')
df2 = pd.read_csv(filepath_or_buffer= '../data/realworld/winequality-red.csv',delimiter=';')
df3 = pd.read_excel(io = '../data/realworld/real_estate.xlsx')



# create dropdown element for selecting the dataset
dataset_dropdown = dcc.Dropdown(['Prostate', 'Wine Quality', 'Real Estate'],
                                clearable = False,
                                # placeholder= 'Select a dataset',
                                id = 'dataset-dropdown',
                                value='Prostate') # set default value to prostate dataset

# create dropdown element for selecting the independent variable
indep_dropdown = dcc.Dropdown(options=df1.columns,
                              clearable=False,
                              # placeholder='Select an independent variable',
                              id='indep-dropdown',
                              value=df1.columns[1])

# create a callback relationship linking database-dropdown to indep-dropdown
@app.callback(Output('indep-dropdown', 'options'),
              Input('dataset-dropdown', 'value'))

def update_indep_dropdown(input_database):
    if input_database == 'Prostate':
        options = df1.columns
    elif input_database == 'Wine Quality':
        options = df2.columns
    elif input_database == 'Real Estate':
        options = df3.columns
    return options


# choose which x and y will be taken, depending on dataset. Maybe another dropdown/slider?
fig = px.scatter(df1,x=df1.columns[3],y=df1.columns[-2])

# create a callback to combine both dropdowns with a scatter plot
@app.callback(Output(component_id='scatter-plot', component_property= 'figure'),
              Input(component_id='dataset-dropdown', component_property='value'), # input 1
              Input(component_id='indep-dropdown', component_property='value') # input 2
)

# Callback function here to select database and independent variable
def update_df(selected_dataset, selected_indep):
    if selected_dataset == 'Prostate':
        df = df1              # choose the right dataframe
        dep = df1.columns[-2] # choose the right dependent variable

    elif selected_dataset == 'Wine Quality':
        df = df2
        dep = df2.columns[-1]
    elif selected_dataset == 'Real Estate':
        df = df3
        dep = df3.columns[-1]

    # create the scatterplot from the selected data
    # TODO: add the regression lines
    fig = px.scatter(df,
                     x=selected_indep,
                     y=dep)
    return fig

# set the layout of the dashboard
app.layout = html.Div([html.H4('Heading'),
                       html.Br(),
                       dataset_dropdown,
                       html.Br(),
                       indep_dropdown,
                       html.Br(),
                       dcc.Graph(id='scatter-plot', figure=fig)]
                      )

if __name__ == '__main__':
    app.run(debug=True,
            port=8000)

# QUICK FINAL CHANGES (some optional)
# TODO: Create flow so that graph name changes depending on dataset
# TODO: Format the scatter plot to look nice
# TODO: