from python_speech_features import mfcc
import wave
import numpy as np
from vad_utils import read_label_from_file
from tqdm import tqdm


def get_mfcc_of_a_wave(wav, frame_size=512, frame_shift=128, sample_rate=16000, num_cep=13):
    frame_count = int((len(wav) - frame_size + frame_shift) / frame_shift)
    mfcc_list = []
    for i in range(frame_count):
        tmp_wav = wav[i * frame_shift:(i + 1) * frame_shift]
        mfcc_list.append(np.squeeze(mfcc(tmp_wav, numcep=num_cep, samplerate=sample_rate)))
    return mfcc_list


def get_mfcc(mfcc_numcep=13, train_or_dev='train'):
    # data is a dictionary
    data = read_label_from_file(path="data/" + train_or_dev + "_label.txt")
    indexes = list(data.keys())
    size = len(indexes)
    mfcc_list = []

    for i in tqdm(range(size)):
        index = '/' + indexes[i] + '.wav'
        f = wave.open("wavs/" + train_or_dev + index, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)

        # data pre-processing
        wave_data_raw = np.frombuffer(str_data, dtype=np.int16)
        wave_data_raw = wave_data_raw - np.mean(wave_data_raw)
        wave_data = wave_data_raw / max(abs(wave_data_raw))

        # get mfcc features and ground truth labels
        mfcc_of_this_wave = get_mfcc_of_a_wave(wave_data, sample_rate=framerate, num_cep=mfcc_numcep)
        mfcc_list.extend(mfcc_of_this_wave)

    mfcc_array = np.array(mfcc_list)
    np.save(train_or_dev + '_mfcc.npy', mfcc_array)
