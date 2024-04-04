import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Load your data
spike_times     = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_times.npy')
spike_clusters  = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_clusters.npy')
FR_SwOn = pd.read_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv',
                                                                        sep=',', usecols=['FR_SwOn'], squeeze=True)
behavior = pd.read_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv')

# Assuming spike_times and FR_SwOn are in samples, and you know the sampling rate
sampling_rate = 30000  # Example: 30 kHz

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#333'}, children=[
    html.H1('Spike Raster Plot', style={'color': '#FFF'}),
    
    dcc.Dropdown(
        id='cluster-dropdown',
        options=[{'label': str(cluster), 'value': cluster} for cluster in np.unique(spike_clusters)],
        value=np.unique(spike_clusters)[0],
        style={'color': '#000'}
    ),
    
    dcc.Graph(id='raster-plot'),
])

@app.callback(
    Output('raster-plot', 'figure'),
    [Input('cluster-dropdown', 'value')]
)
def update_raster(selected_cluster):
    # Filter spikes for the selected cluster
    selected_spike_times = spike_times[spike_clusters == selected_cluster].squeeze()
    
    # Align spikes to the event time and convert to seconds
    aligned_spike_times = (selected_spike_times - FR_SwOn) / sampling_rate
    
    # Generate raster plot
    fig = go.Figure(data=go.Scatter(
        x=aligned_spike_times.flatten(), 
        y=np.zeros_like(aligned_spike_times).flatten(), 
        mode='markers', 
        marker=dict(color='white', size=1)
    ))
    
    fig.update_layout(
        plot_bgcolor='#333',
        paper_bgcolor='#333',
        xaxis=dict(color='white'),
        yaxis=dict(color='white', showticklabels=False),  # Hide y-axis ticks
        title='Cluster ' + str(selected_cluster),
        title_font_color='white'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
