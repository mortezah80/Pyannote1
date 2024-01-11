
import numpy as np
from pydub import AudioSegment
from vad import Vad
import os
from diarization import diarization
import copy
from libs.audio.features import Pretrained
from libs.audio.utils.signal import Binarize
import torchaudio
import yaml

class SpeechProcessingUnit():

    def __init__(self):
        self.fs = 16000
        self.path = ""
        self.args = ""

    def initialize_config(self, threshold):

        path = os.path.dirname(os.path.realpath(__file__))
        self.path = path
        print(path)
        with open(f'{path}/finetuned_models/models/params.yml') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
            self.args = args

        with open(f'{path}/finetuned_models/pipelines/dia_ami/config.yml') as s:
            pipe = yaml.load(s, Loader=yaml.FullLoader)

        pipe['pipeline']['params']['sad_scores'] = os.path.join(path, "finetuned_models/models/sad/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0499.pt")
        pipe['pipeline']['params']['scd_scores'] = os.path.join(path, "finetuned_models/models/scd/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0117.pt")
        pipe['pipeline']['params']['embedding'] = os.path.join(path, "finetuned_models/models/emb/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0135.pt")


        args["speech_turn_segmentation"]["speech_activity_detection"]["onset"] = float(threshold)
        args["speech_turn_segmentation"]["speech_activity_detection"]["offset"] = float(threshold)
        with open(f'{path}/finetuned_models/pipelines/dia_ami/config.yml', "w") as s:
            yaml.dump(pipe, s)

        with open(f'{path}/finetuned_models/models/params.yml', "w") as f:
            yaml.dump(args, f)

    def vad(self, signal, fs, gpu_number):
        """
        
        :param signal:  numpy array of signal data
        :param fs: sample rate
        :return: {"vad_out": vad output(numpy array), "fs": sample rate of vad(int)}
        """

        # signal = signal.astype(np.float64)
        if len(signal.shape) > 1:
            signal = signal[:, 0]

        # if fs != self.fs:
        #     sound = AudioSegment(signal.tobytes(), frame_rate=fs, sample_width= min(4, signal.dtype.itemsize), channels=1)
        #     sound.set_frame_rate(self.fs)
        #     signal = np.frombuffer(sound._data, dtype=np.float16, offset=0)

        vad_unit = Vad(gpu_number)
        vad_out, fs_vad = vad_unit.run(signal, fs)

        result = {}
        result["vad_out"] = vad_out.tolist()
        result['fs'] = fs_vad
        # vad_json = json.dumps

        return result
    def prepare_audio(self,file_dir):
        if (file_dir.split(".")[-1]) == "mp3":
            torchaudio.set_audio_backend("sox_io")
            fs = 16000
            signal, sample_rate = torchaudio.load(file_dir, format="mp3")
            file_duration = len(signal[0]) / sample_rate
            # adjust fs
            adjust_fs = torchaudio.transforms.Resample(sample_rate, fs)
            signal = adjust_fs(signal)
            torchaudio.save('./sound.wav', signal, fs, format="wav")
            pre_file_dir = '/sound.wav'
        else:
            torchaudio.set_audio_backend("sox_io")
            fs = 16000
            signal, sample_rate = torchaudio.load(file_dir, format="wav")
            file_duration = len(signal[0]) / sample_rate
            # adjust fs
            adjust_fs = torchaudio.transforms.Resample(sample_rate, fs)
            signal = adjust_fs(signal)
            torchaudio.save('./sound.wav', signal, fs, format="wav")
            pre_file_dir = './sound.wav'

        return pre_file_dir


    def segmentaion(self, file_dir):

        sad = Pretrained(
            validate_dir=f'{self.path}/finetuned_models/models/sad/train/AMI.SpeakerDiarization.MixHeadset.train/weights')
        print("load success")
        test_file = {'uri': '023_30_M', 'audio': file_dir}
        sad_scores = sad(test_file)
        print(sad_scores)
        sad_args = self.args["speech_turn_segmentation"]["speech_activity_detection"]
        binarize = Binarize(offset=sad_args["offset"], onset=sad_args["onset"], log_scale=True,
                            min_duration_off=sad_args["min_duration_off"], min_duration_on=sad_args["min_duration_on"])

        speech = binarize.apply(sad_scores, dimension=1)

        print(speech)
        print(speech.segments_list_)

        speech_segments = []
        temp_segment = {}
        for i in speech.segments_list_:
            temp_segment["begin"] = i.start
            temp_segment["end"] = i.end
            speech_segments.append(copy.deepcopy(temp_segment))

        return speech_segments


    def speaker_diarization(self, path_to_file, speech_segments, gpu_number):
        """
        
        :param path_to_file: path of audio file
        :param speech_segments: segmens of speech from vad output ex: [{"begin": 0.233, "end": 2.085}, {"begin": 2.869, "end": 4.379}]
        :return: a list of clusters ex: [{"speakers": [{"id": int, "prob": float}, ...], "begin": float, "end": float}, ...]
        """
        clusters = diarization(path_to_file, speech_segments, gpu_number)

        return clusters



    def export(self, sig, fs_sig, vad, fs_vad):

        channels = len(sig.shape)
        if channels == 2:
            # print('WARNING: stereo to mono')
            sig = sig[:, 0]

        vad = np.asarray(vad)
        scale = int(fs_sig / fs_vad)
        vad = np.repeat(vad, scale)
        vad = vad[:len(sig)]

        idx = np.where(vad == 1)
        speech = sig[idx[0]]

        noise = np.delete(sig, idx)

        return speech, noise





if __name__ == '__main__':

    # fs , signal = wav.read("sample.wav")
    # fs, signal = wav.read("t6_speech.wav")
    # [signal, fs] = sf.read("sa1.wav")
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    fs = 16000
    sound = AudioSegment.from_mp3(os.path.join(cur_dir, '00000.mp3'))

    # adjust fs
    if sound.frame_rate != fs:
        sound = sound.set_frame_rate(fs)

    signal = np.frombuffer(sound._data, dtype=np.int16, offset=0)

    # saving signal
    # sound.export('demo16', format='mp3')

    # creating sound object from vector and adjusting fs
    # sound = AudioSegment(signal.tobytes(), frame_rate=fs, sample_width=signal.dtype.itemsize, channels=1)
    # sound = sound.set_frame_rate(16000)
    # signal = np.frombuffer(sound._data, dtype=np.int16, offset=0)


    spu = SpeechProcessingUnit()
    vad_out = spu.vad(signal, fs)
    print(vad_out)

    # speech_segments = [{"begin": 0.233, "end": 2.085}, {"begin": 2.869, "end": 4.379}, {"begin": 4.5, "end": 4.7}]
    spu.speaker_diarization(signal, fs, speech_segments)


    # plt.figure()
    # plt.plot(signal)
    # plt.figure()
    # plt.plot(vad_out["vad_out"])
    # plt.show()
