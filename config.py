'''
I wasted a week of my life making an app that could have been a function call (:
'''
AP      = r'E:\VIV_23058\S10\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0\VIV_23058_S10_g1_t0.imec0.ap.bin'
NCHAN   = 385
ST      = r'E:\VIV_23058\S10\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0\kilosort4\spike_times.npy'
SC      = r'E:\VIV_23058\S10\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0\kilosort4\spike_clusters.npy'
BH      = r'E:\processing_S10\VIV_23058_S10_behavioral_descriptor.csv'
cols    = ['sessionwise_time', 'FR_SwOn', 'FR_StOn', 'HR_SwOn', 'HR_StOn', 'FL_SwOn', 'FL_StOn', 'HL_SwOn', 'HL_StOn']
# In your main app
from config import ST, SC, BH
