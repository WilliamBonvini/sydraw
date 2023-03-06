import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from sydraw import synth

app = Dash(__name__)


app.layout = html.Div([
    dcc.Graph(id="scatter-plot",
              style={'width': '90vh',
                     'height': '90vh',
                     'float': 'left',
                     'show_scale': 'false'}),

    # Number of models
    html.Div([
        html.P("Number of models"),
        dcc.Slider(
            id='nm',
            min=1, max=5, step=1,
            marks={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
            value=2,
        ),

        # Number of points
        html.P("Number of points"),
        dcc.Slider(
                id='n',
                min=100, max=1000, step=100,
                marks={100: '100',
                       200: '200',
                       300: '300',
                       400: '400',
                       500: '500',
                       600: '600',
                       700: '700',
                       800: '800',
                       900: '900',
                       1000: '1000'},
                value=500
            ),

        # Noise
        html.P("Noise percentage"),
        dcc.Slider(
                id='noise_perc',
                min=0, max=0.1, step=0.01,
                marks={0: '0.0',
                       0.01: '0.01',
                       0.02: '0.02',
                       0.03: '0.03',
                       0.04: '0.04',
                       0.05: '0.05',
                       0.06: '0.06',
                       0.07: '0.07',
                       0.08: '0.08',
                       0.09: '0.09',
                       0.1: '0.10'},
                value=0.01
            ),

        # Outliers
        html.P("Outliers percentage"),
        dcc.Slider(
                id='outliers_perc',
                min=0.1, max=1, step=0.10,
                marks={0: '0.0',
                       0.1: '0.1',
                       0.2: '0.2',
                       0.3: '0.3',
                       0.4: '0.4',
                       0.5: '0.5',
                       0.6: '0.6',
                       0.7: '0.7',
                       0.8: '0.8',
                       0.9: '0.9',
                       1: '1.0'},
                value=0.1
            ),
        html.P("Radii range"),
                dcc.RangeSlider(
                        id='radius',
                        min=0.1, max=1.0, step=0.10,
                        marks={0: '0.0',
                               0.1: '0.1',
                               0.2: '0.2',
                               0.3: '0.3',
                               0.4: '0.4',
                               0.5: '0.5',
                               0.6: '0.6',
                               0.7: '0.7',
                               0.8: '0.8',
                               0.9: '0.9',
                               1: '1.0'},
                        value=[0.3, 0.5]
                    )
        ],
    style={'width': '90vh',
           'height': '90vh',
           'float': 'right',
           'padding-top': '10%'})
])


@app.callback(
    Output("scatter-plot", "figure"),
    Input("nm", "value"),
    Input("n", "value"),
    Input("noise_perc", "value"),
    Input("outliers_perc", "value"),
    Input("radius", "value"))
def update_bar_chart(nm, n, noise_perc, outliers_perc, radius):
    sample = synth.circles_sample(
        nm=nm,
        n=n,
        noise_perc=noise_perc,
        outliers_perc=outliers_perc,
        radius=radius,
        homogeneous=False
    )
    df = pd.DataFrame(sample, columns=["x", "y", "label"])
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        range_x=(-2.5, 2.5),
        range_y=(-2.5, 2.5),
    template='simple_white')
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker={'size': 3})

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)

