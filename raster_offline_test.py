import CareyEphys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly
plotly.io.renderers.default= 'browser'
# import plotly.io as pio
# pio.renderers.default = "browser"
import plotly.express as px


sampling_rate = 30000
spike_times = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_times.npy')
spike_times = spike_times.squeeze()/sampling_rate
spike_clusters  = np.load(r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\spike_clusters.npy')
behavior = pd.read_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv')
event_series = 'FR_SwOn'
event_times = behavior.sessionwise_time[behavior[event_series]].values
event_times = event_times[::100]

selected_cluster = 245
selected_spike_times = spike_times[spike_clusters == selected_cluster].squeeze()

plt.figure()
CareyEphys.spikeRaster(selected_spike_times, event_times, time_bounds=[-0.2, 0.2])

##
# manually create a raster from a pandas dataset for plotting with raster

time_bounds=[-0.2, 0.2]
npts = 500
indices = np.linspace(0, event_times.shape[0], npts, dtype=int, endpoint=False) if event_times.shape[0] > npts else range(event_times.shape[0])

df_spikeraster = pd.DataFrame(columns=['event_centred_time', 'y_line'], data=None)
for ii in indices:
    df_this_event = None
    df_this_event = pd.DataFrame(columns=df_spikeraster.columns, data=None)
    these_bounds = [event_times[ii] + time_bounds[0], event_times[ii] + time_bounds[1]]
    these_spikes = selected_spike_times[np.logical_and(selected_spike_times > these_bounds[0], selected_spike_times < these_bounds[1])]
    df_this_event['event_centred_time'] = these_spikes - event_times[ii]
    df_this_event['y_line'] = ii
    df_spikeraster = df_spikeraster.append(df_this_event, ignore_index=True)

## seaborn plot
plt.figure()
sns.scatterplot(data=df_spikeraster, x='event_centred_time', y='y_line', s=1, color='black')

## plotly semistatic plot
fig = px.scatter(data_frame=df_spikeraster, x='event_centred_time', y='y_line')
fig.update_traces(marker_size=3, marker_color='black')
fig.show()

##

