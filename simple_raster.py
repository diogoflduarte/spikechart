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
tmin = 0.2 # s
tmax = 0.2
# Load your data
print('Reading neural data')
spike_times     = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_times.npy')
spike_times = spike_times.squeeze()/sampling_rate
spike_clusters  = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_clusters.npy')
print('Reading behavioral data')
# FR_SwOn = pd.read_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv',
#                                                                         sep=',', usecols=['FR_SwOn'], squeeze=True)
behavior = pd.read_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv')
print('Done')
event_series = 'FR_SwOn'
event_times = behavior.sessionwise_time[behavior[event_series]].values

app.layout = html.Div(style={'backgroundColor': '#333'}, children=[
    html.H1('Spike Raster Plot', style={'color': '#FFF'}),
    dcc.Dropdown(
        id='cluster-dropdown',
        options=[{'label': str(cluster), 'value': cluster} for cluster in np.unique(spike_clusters)],
        value=245,#value=np.unique(spike_clusters)[0],
        style={'color': '#000'}
    ),
    dcc.Graph(id='raster-plot'),
])

@app.callback(
    Output('raster-plot', 'figure'),
    [Input('cluster-dropdown', 'value')]
)
def update_raster(selected_cluster):
    # protections for event time out of bounds of recordings:
    time_bounds = [-tmin, tmax]
    selected_spike_times = spike_times[spike_clusters == selected_cluster].squeeze()
    npts = 500
    indices = np.linspace(0, event_times.shape[0], npts, dtype=int, endpoint=False) \
                if event_times.shape[0] > npts else range(event_times.shape[0])
    df_spikeraster = pd.DataFrame(columns=['event_centred_time (s)', 'y_line'], data=None)
    for ii in indices:
        df_this_event = pd.DataFrame(columns=df_spikeraster.columns, data=None)
        these_bounds = [event_times[ii] + time_bounds[0], event_times[ii] + time_bounds[1]]
        these_spikes = selected_spike_times[np.logical_and(selected_spike_times > these_bounds[0], selected_spike_times < these_bounds[1])]
        df_this_event['event_centred_time (s)'] = these_spikes - event_times[ii]
        df_this_event['y_line'] = ii
        df_spikeraster = df_spikeraster.append(df_this_event, ignore_index=True)

    a = 1

    # Generate raster plot
    fig = px.scatter(data_frame=df_spikeraster, x='event_centred_time (s)', y='y_line')
    fig.update_traces(marker_size=3, marker_color='black')
    fig.update_layout(
        # plot_bgcolor='#333',
        # paper_bgcolor='#333',
        xaxis=dict(color='black'),
        yaxis=dict(color='black', showticklabels=False),  # Hide y-axis ticks
        title='Cluster ' + str(selected_cluster),
        # title_font_color='white'
    )
    return fig
    # img_bytes = fig.to_image(format="png")
    # encoding = b64encode(img_bytes).decode()
    # img_b64 = "data:image/png;base64," + encoding
    # return html.Img(src=img_b64, style={'height': '500px'})

if __name__ == '__main__':
    app.run_server(debug=True)
