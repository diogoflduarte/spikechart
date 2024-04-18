from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd
import base64
import io

TMIN    = -0.2
TMAX    = 0.2
NTRIALS = 500
MSIZE   = 3

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initial dummy data
initial_data = {
    'spike_times': np.array([]),
    'spike_clusters': np.array([]),
    'behavior': pd.DataFrame()
}

app.layout = dbc.Container([
    html.H1('Spike Raster Plot'),
    dcc.Store(id='data-store'),  # Store for holding uploaded data
    dbc.Row([
        dbc.Col(dcc.Upload(
            id='upload-spike-times',
            children=html.Div(['Spike times', html.A('')]),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                   'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                   'textAlign': 'center', 'margin': '10px'},
            multiple=False
        ), width=4),
        dbc.Col(dcc.Upload(
            id='upload-spike-clusters',
            children=html.Div(['Spike clusters', html.A('')]),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                   'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                   'textAlign': 'center', 'margin': '10px'},
            multiple=False
        ), width=4),
        dbc.Col(dcc.Upload(
            id='upload-behavior',
            children=html.Div(['Behavior', html.A('')]),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                   'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                   'textAlign': 'center', 'margin': '10px'},
            multiple=False
        ), width=4)
    ]),
    html.Button('Load Data', id='load-data-button'),
    dcc.Graph(id='raster-plot'),
    html.Label('Cluster:'),
    dcc.Dropdown(                                                                                           # Input 1
        id='cluster-dropdown',
        options=[{'label': str(cluster), 'value': cluster} for cluster in np.unique(initial_data['spike_clusters'])],
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
    html.Label('Behavioral feature'),
    dcc.Dropdown(                                                                                           # Input 5
        id='feature-dropdown',
        options=[{'label': feat, 'value': feat} for feat in list(initial_data['behavior'].columns)],
        value='FR_SwOn',
        style={'width': '20%', 'margin': '10px'}),
])
@app.callback(
    Output('data-store', 'data'),
    Input('load-data-button', 'n_clicks'),
    [State('upload-spike-times', 'contents'),
     State('upload-spike-times', 'filename'),
     State('upload-spike-clusters', 'contents'),
     State('upload-spike-clusters', 'filename'),
     State('upload-behavior', 'contents'),
     State('upload-behavior', 'filename')],
    prevent_initial_callback=True,
)
def update_data_store(n_clicks, spike_times_contents, spike_times_name, spike_clusters_contents, spike_clusters_name, behavior_contents, behavior_name):
    if n_clicks is None:
        raise PreventUpdate

    data = {
        'spike_times': parse_contents(spike_times_contents, spike_times_name),
        'spike_clusters': parse_contents(spike_clusters_contents, spike_clusters_name),
        'behavior': parse_contents(behavior_contents, behavior_name)
    }
    return data

def parse_contents(contents, filename):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.npy'):
            return np.load(io.BytesIO(decoded), allow_pickle=True)
    except Exception as e:
        print(e)
        return None

@app.callback(
    Output('raster-plot', 'figure'),
    [Input('data-store', 'data'),
     Input('cluster-dropdown', 'value'),
     Input('input-tmin', 'value'),
     Input('input-tmax', 'value'),
     Input('input-ntrials', 'value'),
     Input('input-msize', 'value'),
     Input('feature-dropdown', 'value')],
    prevent_initial_call=True
)
def update_raster(data, selected_cluster, tmin, tmax, ntrials, msize, feature):
    spike_times = data['spike_times']
    spike_clusters = data['spike_clusters']
    behavior = data['behavior']

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
