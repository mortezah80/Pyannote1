import os
import timeit
import numpy as np
import audio2numpy as ap
from matplotlib import pyplot as plt
from libs.core import Segment, Timeline, notebook, Annotation
from hub import load
from libs.database.util import load_rttm
from libs.audio.utils.signal import Binarize
from libs.core import SlidingWindowFeature
from diarization import diarization, change_annot
from spu import SpeechProcessingUnit

cur_dir = os.path.dirname(os.path.realpath(__file__))

# cur_file = {'uri': cur_dir + '/test/long_sample', 'audio': cur_dir + '/test/long_sample.wav', 'annotation': load_rttm(cur_dir + '/test/long_sample.rttm')['long_sample']}
# cur_file = {'uri': cur_dir + '/test/long_sample', 'audio': cur_dir + '/test/long_sample.wav'}


start = timeit.default_timer()
waveform, fs = ap.open_audio(cur_dir + '/test/voice.mp3')
# waveform, fs = ap.open_audio(cur_dir + '/test/long_sample.wav')

stop = timeit.default_timer()

print('load audio Time: ', stop - start)

start = timeit.default_timer()
spu = SpeechProcessingUnit()

vad_out = spu.vad(waveform, fs, 0)
stop = timeit.default_timer()

print('VAD Time: ', stop - start)


start = timeit.default_timer()
vad_out = change_annot(np.array(vad_out['vad_out']))
stop = timeit.default_timer()

print('Change Time: ', stop - start)


start = timeit.default_timer()
# annotation, dia, sad, ovl_scores, overlap = diarization(cur_dir + '/test/long_sample.wav', vad_out, 0)
annotation = diarization(cur_dir + '/test/voice.mp3', vad_out, 0)
# annotation = diarization(cur_dir + '/test/long_sample.wav', vad_out, 0)

stop = timeit.default_timer()

print('dia Time: ', stop - start)

# let's visualize the diarization output using pyannote.core visualization API


# # only plot one minute (((i.e. between t=120s and t=180s)
# notebook.crop = Segment(0, 60)
# plot_ready = lambda scores: SlidingWindowFeature(np.exp(scores.data[:, 1:]), scores.sliding_window)
#
# # create a figure with 5 rows with matplotlib
# nrows = 5
# fig, ax = plt.subplots(nrows=nrows, ncols=1)
# fig.set_figwidth(20)
# fig.set_figheight(nrows * 2)
#
# # 1st row: reference annotation
# notebook.plot_annotation(cur_file['annotation'], ax=ax[0], time=False)
# ax[0].text(notebook.crop.start + 0.5, 0.1, 'reference', fontsize=14)
#
# # 2nd row: pipeline output
# notebook.plot_annotation(dia, ax=ax[1], time=False)
# ax[1].text(notebook.crop.start + 0.5, 0.1, 'hypothesis', fontsize=14)
#
# # 3th row: SAD result
# notebook.plot_timeline(sad, ax=ax[2], time=False)
# ax[2].text(notebook.crop.start + 0.5, 0.1, 'speech activity detection', fontsize=14)
#
# # 5th row: OVL raw scores
# notebook.plot_feature(plot_ready(ovl_scores), ax=ax[3], time=False)
# ax[3].text(notebook.crop.start + 0.5, 0.2, 'overlapped speech\ndetection scores', fontsize=14)
# ax[3].set_ylim(-0.1, 1.1)
#
# # 6th row: OVL result
# notebook.plot_timeline(overlap, ax=ax[4], time=False)
# ax[4].text(notebook.crop.start + 0.5, 0.1, 'overlapped speech detection', fontsize=14)
#
# plt.show()
