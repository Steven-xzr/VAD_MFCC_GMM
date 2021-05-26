import wave
import numpy as np
from vad_utils import read_label_from_file
from vad_utils import prediction_to_vad_label
import evaluate
from sklearn.mixture import GaussianMixture as GMM


def main(gmm_components=10,
         filter_length=9,
         train_how_many=1000000):
    # data is a dictionary
    frame_size = 512
    frame_shift = 128
    data = read_label_from_file(path="data/train_label.txt")
    indexes = list(data.keys())
    size = len(indexes)
    train_true_labels = []
    for i in range(size):
        index = '/' + indexes[i] + '.wav'
        f = wave.open("wavs/train" + index, "rb")
        n_frames = f.getparams()[3]
        n_slices = int((n_frames - frame_size + frame_shift) / frame_shift)
        train_label = data[indexes[i]]
        label_pad = np.pad(train_label, (0, np.maximum(n_slices - len(train_label), 0)))
        train_true_labels.extend(label_pad)

    # get_mfcc(train_or_dev='train')
    train_mfcc = np.load("train_mfcc.npy")
    print("Start fitting the GMM model...")
    voice_mfcc = []
    silence_mfcc = []
    voice_label = []
    silence_label = []
    for i in range(train_how_many):
        if train_true_labels[i] == 1:
            voice_mfcc.append(train_mfcc[i])
            voice_label.append(1)
        else:
            silence_mfcc.append(train_mfcc[i])
            silence_label.append(0)
    gmm_voice = GMM(n_components=gmm_components).fit(voice_mfcc, voice_label)
    gmm_silence = GMM(n_components=gmm_components).fit(silence_mfcc, silence_label)
    print("GMM fitting completes.")

    # data is a dictionary
    dev_data = read_label_from_file(path="data/dev_label.txt")
    dev_indexes = list(dev_data.keys())
    dev_size = len(dev_indexes)
    dev_slice_num = np.zeros(dev_size, dtype=np.int16)
    dev_true_labels = []
    file_vad = open("dev_label_task2.txt", "w")
    predict_label_list = []

    # get true labels of dev set
    for i in range(dev_size):
        index = '/' + dev_indexes[i] + '.wav'
        f = wave.open("wavs/dev" + index, "rb")
        n_frames = f.getparams()[3]
        n_slices = int((n_frames - frame_size + frame_shift) / frame_shift)
        dev_slice_num[i] = n_slices
        dev_label = dev_data[dev_indexes[i]]
        label_pad = np.pad(dev_label, (0, np.maximum(n_slices - len(dev_label), 0)))
        dev_true_labels.extend(label_pad)

    # predict
    dev_mfcc = np.load("dev_mfcc.npy")
    predict_voice_score = gmm_voice.score_samples(dev_mfcc)
    predict_silence_score = gmm_silence.score_samples(dev_mfcc)

    # prediction filtered with average
    for j in range(len(dev_mfcc)):
        voice_mean = np.mean(predict_voice_score[j:(j + filter_length)])
        predict_voice_score[j] = voice_mean
        silence_mean = np.mean(predict_silence_score[j:(j + filter_length)])
        predict_silence_score[j] = silence_mean
        if predict_voice_score[j] >= predict_silence_score[j]:
            predict_label_list.append(1)
        else:
            predict_label_list.append(0)

    # delete some too short speeches
    count = 0
    for j in range(len(dev_mfcc)):
        if predict_label_list[j] == 1:
            count += 1
        if predict_label_list[j] == 0 and 7 >= count >= 1:
            for i in range(count):
                predict_label_list[j - count + i] = 0
        if predict_label_list[j] == 0:
            count = 0

    # store the results
    position = 0
    for i in range(dev_size):
        vad_list = prediction_to_vad_label(predict_label_list[position:position+dev_slice_num[i]])
        position += dev_slice_num[i]
        file_vad.write(dev_indexes[i] + " " + vad_list + "\n")
    file_vad.close()

    # calculate accuracy
    arr_pred = np.array(predict_label_list)
    arr_label = np.array(dev_true_labels)
    same_num = arr_label.shape[0] - sum(abs(arr_label - arr_pred))
    print("accuracy:", same_num / arr_label.shape[0])

    # print auc err
    auc, err = evaluate.get_metrics(predict_label_list, dev_true_labels)
    print("auc:", auc)
    print("err:", err)


if __name__ == '__main__':
    main(gmm_components=10, filter_length=9, train_how_many=1000000)
