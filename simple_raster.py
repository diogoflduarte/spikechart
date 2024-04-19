import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from base64 import b64encode
import simpleaudio as sa
from config import *

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
cols = ['sessionwise_time', 'FR_SwOn', 'FR_StOn', 'HR_SwOn', 'HR_StOn', 'FL_SwOn', 'FL_StOn', 'HL_SwOn', 'HL_StOn']

st_file = ST
sc_file = SC
bh_file = BH
apbin   = AP

sampling_rate = 30000.0  # Example: 30 kHz
# Load your data
print('Reading neural data')
spike_times     = np.load(st_file)
spike_times = spike_times.squeeze()/sampling_rate
spike_clusters  = np.load(sc_file)

print('Mapping ap.bin file')
if np.size(np.memmap(apbin, np.uint16, 'r')) % NCHAN is not 0:
    raise ValueError('raw data number of elements not divisible by the default number of channels')
nsamp = int(np.size(np.memmap(apbin, np.int16, 'r'))/385)
dat = np.memmap(apbin, np.int16, 'r', shape=(nsamp, NCHAN))

print('Reading behavioral data')
behavior = pd.read_csv(bh_file, usecols=cols)
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
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div([
                html.Label("Start Time (s):"),
                dcc.Input(id="input-start-time", type="number", value=0, style={'marginRight':'10px'}),
                html.Label("Duration (s):"),
                dcc.Input(id="input-duration", type="number", value=1, style={'marginRight':'10px'}),
                html.Label("Channel:"),
                dcc.Input(id="input-channel", type="number", value=0, style={'marginRight':'10px'}),
                dbc.Button("Play Audio", id="play-audio-button", color="primary"),
            ]), width=12)
        ])
    ]),
    html.Div(id="audio-status", children="Audio Status: Not Playing", style={'color': 'black', 'marginTop': '20px'})
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

@app.callback(
    Output('audio-status', 'style'),
    Output('audio-status', 'children'),
    Input('play-audio-button', 'n_clicks'),
    State('input-start-time', 'value'),
    State('input-duration', 'value'),
    State('input-channel', 'value'),
    prevent_initial_call=True
)
def play_audio(n_clicks, start_time, duration, channel):
    if not callback_context.triggered:
        return {'color': 'black'}, "Audio Status: Not Playing"
    SOUNDRATE = 44100
    sample_rate = int(30000)

    start_index = int(start_time * sample_rate)
    end_index = start_index + int(duration * sample_rate)
    channel = int(channel)

    audio_data = dat[start_index:end_index, channel]
    audio_data = (normalizeTo1(audio_data) * np.iinfo(np.int16).max).astype(np.int16)
    audio_data = np.interp(np.linspace(0, 1, int(duration*SOUNDRATE),   endpoint=False),
                           np.linspace(0, 1, int(duration*sample_rate), endpoint=False),
                           audio_data)
    audio_data = audio_data.astype(np.int16)

    play_obj = sa.play_buffer(audio_data, 1, 2, SOUNDRATE)
    style = {'color': 'green', 'marginTop': '20px'}
    children = "Audio Status: Playing"

    # Wait for audio to finish
    # play_obj.wait_done()

    # Change color to red once finished
    return style, children

def normalizeTo1(a):
    return a/np.max(np.abs(a))


if __name__ == '__main__':
    app.run_server(debug=True)
