import numpy as np
import simpleaudio as sa
import time


def normalizeTo1(a):
    return a/np.max(np.abs(a))
def play_audio(traces, start_time, duration, channel):
    SOUNDRATE = 44100
    sample_rate = int(30000)

    start_index = int(start_time * sample_rate)
    end_index = start_index + int(duration * sample_rate)
    channel = int(channel)

    audio_data = traces[start_index:end_index, channel]
    audio_data = (normalizeTo1(audio_data) * np.iinfo(np.int16).max).astype(np.int16)
    # audio_data = (audio_data * 32767).astype(np.int16)

    # re-sample to 44100
    audio_data = np.interp(np.linspace(0, 1, int(duration*SOUNDRATE),       endpoint=False),
                           np.linspace(0, 1, int(duration*sample_rate), endpoint=False),
                           audio_data)
    audio_data = audio_data.astype(np.int16)

    play_obj = sa.play_buffer(audio_data, 1, 2, SOUNDRATE)
    play_obj.wait_done()


tic = lambda : time.perf_counter()
toc = lambda : tic()
elapsed = lambda x : time.perf_counter() - x

NCHAN = 385

# datloc = r"X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\VIV_23058_S10_g1_t0.imec0.ap.bin"
datloc = r"E:\VIV_23058\S10\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0\VIV_23058_S10_g1_t0.imec0.ap.bin"
# dat = np.memmap(datloc, np.uint16, 'r')
if np.size(np.memmap(datloc, np.uint16, 'r')) % NCHAN is not 0:
    raise ValueError('raw data number of elements not divisible by the default number of channels')

nsamp = int(np.size(np.memmap(datloc, np.int16, 'r'))/385)
dat = np.memmap(datloc, np.int16, 'r', shape=(nsamp, NCHAN))

x = tic()
play_audio(dat, 4000.0, 10, 246)
print(elapsed(x))
