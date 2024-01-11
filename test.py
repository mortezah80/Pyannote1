
import json
import sys

from spu import SpeechProcessingUnit
from pydub import AudioSegment
import numpy as np
from matplotlib import pyplot as plt
import copy

from libs.audio.features import Pretrained
from pydub import AudioSegment
from libs.audio.utils.signal import Binarize
import torchaudio

# fs = 16000
# sound = AudioSegment.from_mp3('/home/aibox/workstation/afshin/data/Sony1Labs-20230131T093458Z-001/Sony1Labs/test/audio/023_30_M.mp3')
#
# #adjust fs
# if sound.frame_rate != fs:
#     sound = sound.set_frame_rate(fs)
#
# signal = np.frombuffer(sound._data, dtype=np.int16, offset=0)


#saving signal
# sound.export('demo16', format='mp3')



#creating sound object from vector and adjusting fs
# sound = AudioSegment(signal.tobytes(), frame_rate=fs, sample_width=signal.dtype.itemsize, channels=1)
# sound = sound.set_frame_rate(16000)
# signal = np.frombuffer(sound._data, dtype=np.int16, offset=0)

# spu = SpeechProcessingUnit()
#
# vad_out = spu.vad(signal, fs, gpu_number=0)
#
# plt.figure()
# plt.plot(signal)
# plt.plot(np.repeat(np.array(vad_out["vad_out"]), 160))
#
# thresh = 0.2
# vad_pred = np.round(np.array(vad_out["vad_out"]) - thresh + 0.5)

# json.dump(vad_out, "vad_out.json")

# speech, noise = spu.export(signal, fs, vad_pred, vad_out["fs"])

# #saving speech and noise
# sound_speech = AudioSegment(speech.tobytes(), frame_rate=fs, sample_width=speech.dtype.itemsize, channels=1)
# sound_speech.export('speech', format='mp3')
#
# sound_noise = AudioSegment(noise.tobytes(), frame_rate=fs, sample_width=noise.dtype.itemsize, channels=1)
# sound_noise.export('noise', format='mp3')

file_dir = "/media/aibox/Mehdi-hard/1hour_audio/sound.wav"



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
    gpu_number = 0

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
            gpu_number = int(arg)

    task = "segmentation_diarization"

    file_dir = file_path

    # if task == "segmentation":
    #
    #     if (file_dir.split(".")[-1]) == "mp3":
    #         torchaudio.set_audio_backend("sox_io")
    #         fs = 16000
    #         signal, sample_rate = torchaudio.load(file_dir, format="mp3")
    #         file_duration = len(signal[0]) / sample_rate
    #         # adjust fs
    #         adjust_fs = torchaudio.transforms.Resample(sample_rate, fs)
    #         signal = adjust_fs(signal)
    #         torchaudio.save('./sound.wav', signal, fs, format="wav")
    #         file_dir = '/sound.wav'
    #     else:
    #         torchaudio.set_audio_backend("sox_io")
    #         fs = 16000
    #         signal, sample_rate = torchaudio.load(file_dir, format="wav")
    #         file_duration = len(signal[0]) / sample_rate
    #         # adjust fs
    #         adjust_fs = torchaudio.transforms.Resample(sample_rate, fs)
    #         signal = adjust_fs(signal)
    #         torchaudio.save('./sound.wav', signal, fs, format="wav")
    #         file_dir = './sound.wav'
    #
    #
    #
    # #
    #     sad = Pretrained(validate_dir='/media/aibox/Mehdi-hard/workstation/vad_gpu_worker_sony-main_ours/external_apps/ai_unit/finetuned_models/models/sad/train/AMI.SpeakerDiarization.MixHeadset.train/weights')
    #     print("load success")
    #     test_file = {'uri': '023_30_M', 'audio': file_dir}
    #     sad_scores = sad(test_file)
    #     print(sad_scores)
    #
    #     binarize = Binarize(offset=0.57, onset=0.57, log_scale=True,
    #                         min_duration_off=0.6315121069334447, min_duration_on=0.0007366523493967721)
    #
    #     speech = binarize.apply(sad_scores, dimension=1)
    #
    #     print(speech)
    #     print(speech.segments_list_)
    #
    #     speech_segments = []
    #     temp_segment = {}
    #     for i in speech.segments_list_:
    #         temp_segment["begin"] = i.start
    #         temp_segment["end"] = i.end
    #         speech_segments.append(copy.deepcopy(temp_segment))
    #
    #     print(speech_segments)

    # with open("annot.txt" , "w") as f:
    #     for i in speech.segments_list_ :
    #         f.write(str(i.start))
    #         f.write(" ")
    #         f.write(str(i.end))
    #         f.write("\n")

    spu = SpeechProcessingUnit()
    spu.initialize_config(threshold)
    prepare_file_dir = spu.prepare_audio(file_dir)

    if task == "segmentation":
        speech_segments = spu.segmentaion(prepare_file_dir)
        print(speech_segments)
    if task == "diarization":

        clusters = spu.speaker_diarization(file_dir, None, gpu_number)
        print(clusters)
        output_string = json.dumps([speech_segments, clusters], indent=4)
        print(output_string)

    if task == "segmentation_diarization":
        speech_segments = spu.segmentaion(prepare_file_dir)
        print(speech_segments)

        clusters = spu.speaker_diarization(file_dir, speech_segments, gpu_number)
        print(clusters)

        output_string = json.dumps([speech_segments, clusters], indent=4)
        print(output_string)
        jsonFile = open("data7.json", "w")
        jsonFile.write(output_string)
        jsonFile.close()
    # if task == "segmentaion":









    # speech_segments = [{"begin": 0.233, "end": 1.2}, {"begin": 2.869, "end": 4.379}, {"begin": 4.5, "end": 4.7}]





    # audio = open("audio.txt", "a")
    # audio_noise = open("audio_noise.txt", "a")
    #
    # # audio_noise = AudioSegment.empty()
    # sound = AudioSegment.from_file("/home/aibox/workstation/afshin/data/Sony1Labs-20230131T093458Z-001/Sony1Labs/test/audio1/023_30_M.wav", "wav")
    #
    #
    # EndTime = 0
    #
    # for i in speech_segments:
    #
    #     StartTime = i["begin"]
    #
    #     audio_noise.write(str(EndTime))
    #     audio_noise.write(" ")
    #     audio_noise.write(str(StartTime))
    #     audio_noise.write("\n")
    #
    #     EndTime = i["end"]
    #
    #     audio.write(str(StartTime))
    #     audio.write(" ")
    #     audio.write(str(EndTime))
    #     audio.write("\n")
    #
    # audio.close()
    # audio_noise.close()
    #
    # # print(audio)


# audio.export("/home/aibox/home/aibox/workstation/afshin/voicefactory-SpeechProcessingUnit-master/mori_test1/audio.wav", format="wav")
# audio_noise.export("/home/aibox/home/aibox/workstation/afshin/voicefactory-SpeechProcessingUnit-master/mori_test/audio_noise.wav", format="wav")


