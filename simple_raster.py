import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from base64 import b64encode
import base64
import io
import sys

# Initialize the Dash app
app = dash.Dash(__name__)

sampling_rate = 30000.0  # Example: 30 kHz
# Load your data
# print('Reading neural data')
# spike_times     = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_times.npy')
# spike_times = spike_times.squeeze()/sampling_rate
# spike_clusters  = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_clusters.npy')
# print('Reading behavioral data')
# behavior = pd.read_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv')
feat = 'FR_SwOn'
print('Done')
TMIN    = -0.2
TMAX    = 0.2
NTRIALS = 500
MSIZE   = 3

# placeholder for initialization
spike_clusters = np.arange(3)
behavior = pd.DataFrame(data=None, columns=['empty'])
spike_times = np.arange(3)

app.layout = html.Div(children=[
    html.H1('Spike Raster Plot'),
    dbc.Container(
        dbc.Row(
            [dbc.Col(
                dcc.Upload(
                    id='upload-spike-times',
                    children=html.Div(['Drop SPIKETIMES or ', html.A('Select Files')]),
                    style={
                        'width': '20%', 'height': '40px', 'lineHeight': '40px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px'
                    },
                    multiple=False
                )
            ),
            dbc.Col(
                dcc.Upload(
                    id='upload-spike-clusters',
                    children=html.Div(['Drop SPIKECLUSTERS or ', html.A('Select Files')]),
                    style={
                        'width': '20%', 'height': '40px', 'lineHeight': '40px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px'
                    },
                    multiple=False
                )
            ),
            dbc.Col(
                dcc.Upload(
                    id='upload-behavior',
                    children=html.Div(['Drop BEHAVIOR or ', html.A('Select Files')]),
                    style={
                        'width': '20%', 'height': '40px', 'lineHeight': '40px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px'
                    },
                    multiple=False
                )
            )]
        )
    ),
    html.Button('Load Data', id='load-data-button'),
    dcc.Graph(id='raster-plot'),
    html.Label('Cluster:'),
    dcc.Dropdown(                                                                                           # Input 1
        id='cluster-dropdown',
        options=[{'label': str(cluster), 'value': cluster} for cluster in np.unique(spike_clusters)],
        value=245,
        style={'width': '20%', 'margin': '10px'}
    ),
    html.Label('tmin:'),
    dcc.Input(id='input-tmin', type='number', value=TMIN),                                                  # Input 2
    html.Label('tmax:'),
    dcc.Input(id='input-tmax', type='number', value=TMAX),                                                  # Input 3
    html.Label('trials:'),
    dcc.Input(id='input-ntrials', type='number', value=NTRIALS),                                            # Input 4
    html.Label('markersize:'),
    dcc.Input(id='input-msize', type='number', value=MSIZE),                                                # Input 4
    dcc.Dropdown(                                                                                           # Input 5
        id='feature-dropdown',
        options=[{'label': feat, 'value': feat} for feat in list(behavior.columns)],
        value='FR_SwOn',
        style={'width': '20%', 'margin': '10px'}
    ),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'npy' in filename:
            # Assume that the user uploaded a numpy file
            df = np.load(io.BytesIO(decoded), allow_pickle=True)
        else:
            return html.Div([
                'Unsupported file type: {}'.format(filename)
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

@app.callback(
    Output('raster-plot', 'figure'),
    [Input('load-data-button', 'n_clicks')],
    [State('upload-spike-times', 'contents'),
     State('upload-spike-times', 'filename'),
     State('upload-spike-clusters', 'contents'),
     State('upload-spike-clusters', 'filename'),
     State('upload-behavior', 'contents'),
     State('upload-behavior', 'filename')],
    prevent_initial_call='initial_duplicate'
)
def update_output(n_clicks, spike_times_contents, spike_times_name, spike_clusters_contents, spike_clusters_name,
                  behavior_contents, behavior_name):
    if n_clicks is None:
        raise PreventUpdate
    print('0')
    spike_times = parse_contents(spike_times_contents, spike_times_name, None)
    print('1')
    spike_clusters = parse_contents(spike_clusters_contents, spike_clusters_name, None)
    print('2')
    behavior = parse_contents(behavior_contents, behavior_name, None)
    print('3')

    # Assuming behavior has a column 'FR_SwOn' and appropriate time column
    # Generate the figure based on the data
    # This is a placeholder; you'll need to adapt it to how you process and visualize your data
    this_clu = 0
    fig = update_raster(this_clu, TMIN, TMAX, NTRIALS, MSIZE, 'FR_SwOn')
    return fig
@app.callback(
    Output('raster-plot', 'figure'),
    [Input('cluster-dropdown', 'value'),
     Input('input-tmin', 'value'),
     Input('input-tmax', 'value'),
     Input('input-ntrials', 'value'),
     Input('input-msize', 'value'),
     Input('feature-dropdown', 'value')],
    prevent_initial_call=False
)
def update_raster(selected_cluster, tmin, tmax, ntrials, msize, feature):
    event_series = feature
    event_times = behavior.sessionwise_time[behavior[event_series]].values
    time_bounds = [tmin, tmax]
    selected_spike_times = spike_times[spike_clusters == selected_cluster].squeeze()
    npts = ntrials
    indices = np.linspace(0, event_times.shape[0], npts, dtype=int, endpoint=False) \
                if event_times.shape[0] > npts else range(event_times.shape[0])
    ntrials = event_times.shape[0]
    df_spikeraster = pd.DataFrame(columns=['event_centred_time (s)', 'y_line'], data=None)
    for ii in indices:
        df_this_event = pd.DataFrame(columns=df_spikeraster.columns, data=None)
        these_bounds = [event_times[ii] + time_bounds[0], event_times[ii] + time_bounds[1]]
        these_spikes = selected_spike_times[np.logical_and(selected_spike_times > these_bounds[0],
                                                           selected_spike_times < these_bounds[1])]
        df_this_event['event_centred_time (s)'] = these_spikes - event_times[ii]
        df_this_event['y_line'] = ii
        df_spikeraster = df_spikeraster.append(df_this_event, ignore_index=True)

    # Generate raster plot
    fig = px.scatter(data_frame=df_spikeraster, x='event_centred_time (s)', y='y_line')
    fig.update_traces(marker_size=msize, marker_color='black')
    fig.update_layout(
        xaxis=dict(color='black'),
        yaxis=dict(color='black', showticklabels=False),  # Hide y-axis ticks
        title='Cluster ' + str(selected_cluster),
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(host="0.0.0.0", port=8050)