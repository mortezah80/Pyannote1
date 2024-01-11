import json
from spu import SpeechProcessingUnit
from pydub import AudioSegment
import numpy as np

# [signal, fs] = sf.read("../../uploads/voices/TZDC4W-s21no-5XONjEt-p6uH4aQC-9GJuEP4KbMobh6Ugw-RTO4_c4t2o0i/segments/00001.mp3")
# vad_out = spu.vad(signal, fs)
# print(vad_out)

# speech_segments = [{"begin": 0.233, "end": 2.085}, {"begin": 2.869, "end": 4.379}, {"begin": 4.5, "end": 4.7}]
# clusters = spu.speaker_diarization(signal, fs, speech_segments)
# print(clusters)

def float_to_byte(i):
    return int(i * 255)

def create_speech_none_speech_segments(threshold, vad, fs = 100):
    i = 0
    threshold = threshold * 255
    vad_len = len(vad)
    segments = []
    begin = 0
    is_speech = (vad[0] > threshold)
    while i + 1 < vad_len:
        i += 1
        if (vad[i] > threshold) != is_speech:
            end = i / fs
            if is_speech:
                segments.append({"begin": begin, "end": end})

            begin = end
            is_speech = not(is_speech)

    if is_speech:
        end = i / fs
        segments.append({"begin": begin, "end": end})


    return segments


def process_speakers(file_path, vad_path, save_path, threshold, gpu_number):
#     speech_segments = None
#     with open("speech_segments.json") as file:
#         speech_segments = json.loads(o.read())
#
#     spu = SpeechProcessingUnit()
#
#     speech_segments = []
#     with open(segment_path, "r") as file:
#         file.write(out)
#
#     [signal, fs] = sf.read(file_path)
#     clusters = spu.speaker_diarization(signal, fs, speech_segments)
#
#     with open(save_path, "w") as file:
#         file.write(json.dumps(clusters)

#     speech_segments = None
#     with open(segment_path) as file:
#         speech_segments = json.loads(o.read())
#
#     speech_segments = []
#     with open(segment_path, "r") as file:
#         file.write(out)

    vad = []
    with open(vad_path, "rb") as file:
        vad = file.read()

    speech_segments = create_speech_none_speech_segments(threshold, vad)

    # fs = 16000
    # sound = AudioSegment.from_mp3(file_path)

    #adjust fs
    # if sound.frame_rate != fs:
    #     sound = sound.set_frame_rate(fs)

    # signal = np.frombuffer(sound._data, dtype=np.int16, offset=0)

    spu = SpeechProcessingUnit()
    clusters = spu.speaker_diarization(file_path, speech_segments, gpu_number)

    with open(save_path, "w") as file:
        file.write(json.dumps(clusters))


import sys

if __name__ == "__main__":
    file_path_index = -1
    save_path_index = -1
    vad_path_index = -1
    threshold_index = -1
    gpu_number_index = -1

    threshold = 0.5
    save_path = ""
    file_path = ""
    vad_path = ""
    gpu_number = None

    for i, arg in enumerate(sys.argv):
        if arg == "file_path":
            file_path_index = i + 1

        elif arg == "save_path":
            save_path_index = i + 1

        elif arg == "vad_path":
            vad_path_index = i + 1

        elif arg == "threshold":
            threshold_index = i + 1

        elif arg == 'gpu_number':
            gpu_number_index = i + 1

        elif vad_path_index == i:
            vad_path = arg

        elif file_path_index == i:
            file_path = arg

        elif save_path_index == i:
            save_path = arg

        elif threshold_index == i:
            threshold = arg

        elif gpu_number_index == i:
            gpu_number = arg


    process_speakers(file_path, vad_path, save_path, float(threshold), gpu_number)

