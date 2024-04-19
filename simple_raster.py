import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from base64 import b64encode

# Initialize the Dash app
app = dash.Dash(__name__)

sampling_rate = 30000.0  # Example: 30 kHz
# Load your data
print('Reading neural data')
spike_times     = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_times.npy')
spike_times = spike_times.squeeze()/sampling_rate
spike_clusters  = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_clusters.npy')
print('Reading behavioral data')
behavior = pd.read_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv')
feat = 'FR_SwOn'
print('Done')

app.layout = html.Div(children=[
    html.H1('Spike Raster Plot'),
    html.Label('Cluster:'),
    dcc.Dropdown(                                                                                           # Input 1
        id='cluster-dropdown',
        options=[{'label': str(cluster), 'value': cluster} for cluster in np.unique(spike_clusters)],
        value=245
    ),
    html.Label('tmin:'),
    dcc.Input(id='input-tmin', type='number', value=-0.2),                                                  # Input 2
    html.Label('tmax:'),
    dcc.Input(id='input-tmax', type='number', value=0.2),                                                   # Input 3
    html.Label('trials:'),
    dcc.Input(id='input-ntrials', type='number', value=500),                                                # Input 4
    html.Label('markersize:'),
    dcc.Input(id='input-msize', type='number', value=3),                                                    # Input 4
    dcc.Dropdown(                                                                                           # Input 5
        id='feature-dropdown',
        options=[{'label': feat, 'value': feat} for feat in list(behavior.columns)],
        value='FR_SwOn'
    ),
    dcc.Graph(id='raster-plot'),
])

@app.callback(
    Output('raster-plot', 'figure'),
    [Input('cluster-dropdown', 'value'),
     Input('input-tmin', 'value'),
     Input('input-tmax', 'value'),
     Input('input-ntrials', 'value'),
     Input('input-msize', 'value'),
     Input('feature-dropdown', 'value')]
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
    # img_bytes = fig.to_image(format="png")
    # encoding = b64encode(img_bytes).decode()
    # img_b64 = "data:image/png;base64," + encoding
    # return html.Img(src=img_b64, style={'height': '500px'})

if __name__ == '__main__':
    app.run_server(debug=True)
