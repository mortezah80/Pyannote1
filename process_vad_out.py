from spu import SpeechProcessingUnit
import soundfile as sf
from pydub import AudioSegment
import numpy as np

# [signal, fs] = sf.read("../../uploads/voices/TZDC4W-s21no-5XONjEt-p6uH4aQC-9GJuEP4KbMobh6Ugw-RTO4_c4t2o0i/segments/00001.mp3")
# vad_out = spu.vad(signal, fs)
# print(vad_out)

# speech_segments = [{"begin": 0.233, "end": 2.085}, {"begin": 2.869, "end": 4.379}, {"begin": 4.5, "end": 4.7}]
# clusters = spu.speaker_diarization(signal, fs, speech_segments)
# print(clusters)

def float_to_byte(i):
    return int(i * 255) if i > 0 else 0

def process_vad_out(file_path, save_path, gpu_number):

    fs = 16000
    sound = AudioSegment.from_mp3(file_path)
    #adjust fs
    if sound.frame_rate != fs:
        sound = sound.set_frame_rate(fs)
    signal = np.frombuffer(sound._data, dtype=np.int16, offset=0)
    spu = SpeechProcessingUnit()
    vad = spu.vad(signal, fs, gpu_number)
    out = bytearray(map(float_to_byte, vad["vad_out"]))
    with open(save_path, "wb") as file:
        file.write(out)


    # spu = SpeechProcessingUnit()
    # [signal, fs] = sf.read(file_path)
    # vad = spu.vad(signal, fs)

    # out = bytearray(map(float_to_byte, vad["vad_out"]))
    # with open(save_path, "wb") as file:
    #     file.write(out)


import sys

if __name__ == "__main__":
    file_path_index = -1
    save_path_index = -1
    gpu_number_index = -1

    gpu_number = None
    save_path = ""
    file_path = ""

    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if arg == "file_path":
            file_path_index = i + 1
        elif arg == "save_path":
            save_path_index = i + 1
        elif arg == 'gpu_number':
            gpu_number_index = i +1
        elif file_path_index == i:
            file_path = arg
        elif save_path_index == i:
            save_path = arg
        elif gpu_number_index == i:
            gpu_number = arg

    process_vad_out(file_path, save_path, gpu_number)

