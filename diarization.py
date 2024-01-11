import os
import numpy as np
from libs.core import Segment, Timeline, Annotation
from hub import load
from libs.audio.utils.signal import Binarize


def change_annot(nums):
    # applying the threshold (0.5)
    nums = np.array(nums.round())

    # extracting speech index
    nums = np.where(nums == 1)[0]
    nums = sorted(set(nums))

    #extracting speech groups
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    lists = list(zip(edges, edges))

    # converting to annotation (sad style)
    annot = []
    for i in lists:
        annot.append({"begin": i[0] / 100, "end": (i[1] + 1) / 100})

    return annot

def diarization(path_to_file, my_sad=None, gpu_number=None):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    cur_file = {'uri': path_to_file.split('.')[0], 'audio': path_to_file}
    device = 'cuda:' + str(gpu_number)
    # loading models and pipelines

    ovl = load(cur_dir, 'ovl', source='local', device=device)
    pipeline = load(cur_dir, 'dia_ami', source='local', device=device)

    # converting sad output
    if my_sad is not None:
        segment = []
        for seg in my_sad:
            segment.append(Segment(seg['begin'], seg['end']))

        sad = Timeline(segments=segment, uri=cur_file['uri'])
    # generating sad in case we don't provide it:
    else:
        sad_pipeline = load(cur_dir, 'sad', source='local', device=device)
        sad_scores = sad_pipeline(cur_file)

        binarize = Binarize(offset=0.52, onset=0.52, log_scale=True,
                        min_duration_off=0.1, min_duration_on=0.1)

        # speech regions (as `pyannote.core.Timeline` instance)
        sad = binarize.apply(sad_scores, dimension=1)


    # Overlap regions
    ovl_scores = ovl(cur_file)

    # binarize raw OVL scores
    # NOTE: both onset/offset values were tuned on AMI dataset.
    # you might need to use different values for better results.
    binarize = Binarize(offset=0.5040963812660672, onset=0.5040963812660672, log_scale=True,
                        min_duration_off=0.1, min_duration_on=0.1)

    # overlapped speech regions (as `pyannote.core.Timeline` instance)
    overlap = binarize.apply(ovl_scores, dimension=1)


    # omitting overlapped segments (will be added after diarization)
    sad_ovl = sad.union(overlap).segmentation()
    for seg in sad_ovl.crop(sad_ovl.crop(overlap, mode='strict'), mode='strict'):
        sad_ovl.remove(seg)

    # diarization
    dia, distances = pipeline(cur_file, sad_ovl)

    # returning overlapped segments

    rev_ann = Annotation()
    for seg in overlap:
        rev_ann[seg, '_'] = 'overlapped'

    dia.update(rev_ann)

    ann_ins = list(dia.itertracks(yield_label=True))
    labels = dia.labels()

    # calcualte probabilities from distances to clusters
    probabilities = []
    for l in distances:
        probabilities.append(1 / np.array(l) / np.sum(1 / np.array(l)))

    p = list(np.zeros(len(labels)))
    for anno in ann_ins:
        if anno[2] == 'overlapped':
            probabilities.insert(ann_ins.index(anno), p)


    # changing annotation format
    changed_ann = []
    for ann in ann_ins:
        if ann[2] == 'overlapped':
            track = {"speakers": [{"id": 0, "prob": 100}], "begin": ann[0].start,
                     "end": ann[0].end}
            for i in range(1, len(labels) + 1):
                track['speakers'].append({"id": i, "prob": 0})
        else:
            probs = list(probabilities[ann_ins.index(ann)])
            track = {"speakers": [], "begin": ann[0].start, "end": ann[0].end}
            for i in sorted(probs, reverse=True):
                track['speakers'].append({'id': probs.index(i) + 1, 'prob': i})

            track['speakers'].append({"id": 0, "prob": 0})

        changed_ann.append(track)

    return changed_ann #, dia, sad, ovl_scores, overlap
