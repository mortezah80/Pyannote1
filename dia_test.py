import os
import numpy as np
from matplotlib import pyplot as plt
from libs.core import Segment, Timeline, notebook, Annotation
from hub import load
from libs.database.util import load_rttm
from libs.audio.utils.signal import Binarize
from libs.core import SlidingWindowFeature
from diarization import diarization

cur_dir = os.path.dirname(os.path.realpath(__file__))
# cur_file = {'uri': cur_dir + '/test/sample', 'audio': cur_dir + '/test/sample.wav', 'annotation': load_rttm(cur_dir + '/test/sample.rttm')['sample']}
# cur_file = {'uri': cur_dir + '/test/ovl_sample', 'audio': cur_dir + '/test/ovl_sample.wav', 'annotation': load_rttm(cur_dir + '/test/ovl_sample.rttm')['ovl_sample']}
# cur_file = {'uri': cur_dir + '/test/EN2001a.Mix-Headset', 'audio': cur_dir + '/test/EN2001a.Mix-Headset.wav', 'annotation': load_rttm(cur_dir + '/test/MixHeadset.train.rttm')['EN2001a.Mix-Headset']}
#
# ovl = load(cur_dir, 'ovl', source='local')
# pipeline = load(cur_dir, 'dia_ami', source='local')
#
# # Overlap regions
# ovl_scores = ovl(cur_file)
#
# # binarize raw OVL scores
# # NOTE: both onset/offset values were tuned on AMI dataset.
# # you might need to use different values for better results.
#
# binarize = Binarize(offset=0.55, onset=0.55, log_scale=True,
#                     min_duration_off=0.1, min_duration_on=0.1)
# overlap = binarize.apply(ovl_scores, dimension=1)
#
# diarization, distances = pipeline(cur_file)
# print(list(diarization.itertracks(yield_label='True')))
# print(distances)


#
sample = [{"begin": 0.233, "end": 2.085}, {"begin": 2.869, "end": 4.379}, {"begin": 5.2, "end": 8.7}, {"begin": 10, "end": 15}]
d = diarization('/home/demo/Desktop/self/speaker_diarization_module (another copy)/test/ovl_sample.wav')

print(d)
# let's visualize the diarization output using libs.core visualization API


# # only plot one minute (between t=120s and t=180s)
# notebook.crop = Segment(0, 60)
# plot_ready = lambda scores: SlidingWindowFeature(np.exp(scores.data[:, 1:]), scores.sliding_window)
#
# # create a figure with 4 rows with matplotlib
# nrows = 4
# fig, ax = plt.subplots(nrows=nrows, ncols=1)
# fig.set_figwidth(20)
# fig.set_figheight(nrows * 2)
#
# # 1st row: reference annotation
# notebook.plot_annotation(cur_file['annotation'], ax=ax[0], time=False)
# ax[0].text(notebook.crop.start + 0.5, 0.1, 'reference', fontsize=14)
#
# # 2nd row: pipeline output
# notebook.plot_annotation(diarization, ax=ax[1], time=False)
# ax[1].text(notebook.crop.start + 0.5, 0.1, 'hypothesis', fontsize=14)
#
#
# # 3th row: OVL raw scores
# notebook.plot_feature(plot_ready(ovl_scores), ax=ax[2], time=False)
# ax[2].text(notebook.crop.start + 0.5, 0.2, 'overlapped speech\ndetection scores', fontsize=14)
# ax[2].set_ylim(-0.1, 1.1)
#
# # 4th row: OVL result
# notebook.plot_timeline(overlap, ax=ax[3], time=False)
# ax[3].text(notebook.crop.start + 0.5, 0.1, 'overlapped speech detection', fontsize=14)
#
# plt.show()
