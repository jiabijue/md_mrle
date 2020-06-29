import csv
import json
import os

import dcase_util
import librosa
import numpy as np
import pandas as pd
import sed_eval
import torch
import torch.utils.data as data

import config as cfg


# ---------- Basic ----------
def load_json(filename):
    with open(filename) as f:
        json_data = json.load(f)
    return json_data


def log(file, exp, setting, epoch_stop_at, test_loss,
        mud_s_metrics, mud_e_metrics, mrle_s_metrics, mrle_e_metrics):
    if not os.path.isfile(file):
        with open(file, "w+", newline='') as f:
            csv_f = csv.writer(f)
            head = ["exp", "anno_dir", "sampling_rate", "n_mels", "fft_windows_size", "hop_length",
                    "model_cfg", "batch_size", "lr", "epoch_stop_at", "test_loss",
                    "mud:acc", "mud:s_all_f", "mud:s_m_f", "mud:s_nom_f", "mud:e_all_f", "mud:e_m_f", "mud:e_nom_f",
                    "mrle:acc", "mrle:s_all_f", "mrle:s_fg_f", "mrle:s_bg_f", "mrle:s_nom_f",
                    "mrle:e_all_f", "mrle:e_fg_f", "mrle:e_bg_f", "mrle:e_nom_f"]
            csv_f.writerow(head)

    # Append new log
    model_cfg = setting["model"]
    # #music detection task#
    mud_acc = mud_s_metrics.overall_accuracy()['accuracy']
    mud_s_all_f = mud_s_metrics.overall_f_measure()['f_measure']
    mud_s_m_f = mud_s_metrics.class_wise_f_measure('music')['f_measure']
    mud_s_nom_f = mud_s_metrics.class_wise_f_measure('no-music')['f_measure']
    mud_e_all_f = mud_e_metrics.overall_f_measure()['f_measure']
    mud_e_m_f = mud_e_metrics.class_wise_f_measure('music')['f_measure']
    mud_e_nom_f = mud_e_metrics.class_wise_f_measure('no-music')['f_measure']

    # #music relative loudness estimation task#
    mrle_acc = mrle_s_metrics.overall_accuracy()['accuracy']
    mrle_s_all_f = mrle_s_metrics.overall_f_measure()['f_measure']
    mrle_s_fg_f = mrle_s_metrics.class_wise_f_measure('fg-music')['f_measure']
    mrle_s_bg_f = mrle_s_metrics.class_wise_f_measure('bg-music')['f_measure']
    mrle_s_nom_f = mrle_s_metrics.class_wise_f_measure('no-music')['f_measure']
    mrle_e_all_f = mrle_e_metrics.overall_f_measure()['f_measure']
    mrle_e_fg_f = mrle_e_metrics.class_wise_f_measure('fg-music')['f_measure']
    mrle_e_bg_f = mrle_e_metrics.class_wise_f_measure('bg-music')['f_measure']
    mrle_e_nom_f = mrle_e_metrics.class_wise_f_measure('no-music')['f_measure']

    with open(file, "a+", newline='') as f:
        csv_f = csv.writer(f)
        row_data = [exp, cfg.ANNO_DIR, cfg.SAMPLING_RATE, cfg.N_MELS, cfg.FFT_WINDOW_SIZE, cfg.HOP_LENGTH,
                    model_cfg, setting["batch_size"], setting["lr"], epoch_stop_at, test_loss,
                    mud_acc, mud_s_all_f, mud_s_m_f, mud_s_nom_f, mud_e_all_f, mud_e_m_f, mud_e_nom_f,
                    mrle_acc, mrle_s_all_f, mrle_s_fg_f, mrle_s_bg_f, mrle_s_nom_f,
                    mrle_e_all_f, mrle_e_fg_f, mrle_e_bg_f, mrle_e_nom_f]
        csv_f.writerow(row_data)


# ----------  Data ----------
## Split data for train, validation, test
def get_split_list(idx):
    df = pd.read_csv(cfg.SPLITS_CSV, sep=',', usecols=[0, 1])
    split_fecha = df.set_index('split')
    split_df = split_fecha.loc[idx]
    split_list = split_df['file_name'].values.tolist()
    return split_list


## Data Augmentation
# TODO


## Pre-processing
def get_log_melspectrogram(audio_file):  # for `Parameters set 1`
    """ computed from audio at `SAMPLING_RATE` Hz and 16 bits per sample,
        with `N_MELS` frequency values for each frame.
        Frames and hop sizes are `FFT_WINDOW_SIZE` and `HOP_LENGTH` samples, respectively.
    """
    samples, _ = librosa.load(audio_file, cfg.SAMPLING_RATE)
    # make a mel-scaled power (energy-squared) spectrogram.
    mels = librosa.feature.melspectrogram(samples, cfg.SAMPLING_RATE,
                                          n_mels=cfg.N_MELS,
                                          n_fft=cfg.FFT_WINDOW_SIZE,
                                          hop_length=cfg.HOP_LENGTH)
    # convert to log scale (dB). use the peak power as reference.
    log_mels = librosa.amplitude_to_db(mels, ref=np.max)
    return log_mels


# def get_scaled_mel_bands(audio_file):  # for `Parameters set 2`
#     """ computed from audio at `SAMPLING_RATE` Hz and 16 bits per sample,
#         with `N_MELS` frequency values for each frame.
#         Frames and hop sizes are `FFT_WINDOW_SIZE` and `HOP_LENGTH` samples, respectively.
#     """
#     samples, _ = librosa.load(audio_file, cfg.SAMPLING_RATE)
#     spec = np.abs(librosa.stft(samples, cfg.FFT_WINDOW_SIZE, cfg.HOP_LENGTH)) ** 2
#     ft = librosa.filters.mel(cfg.SAMPLING_RATE, cfg.FFT_WINDOW_SIZE,
#                              cfg.N_MELS, cfg.F_MIN, cfg.F_MAX)
#     bands = np.dot(ft, spec)
#     return librosa.core.power_to_db(bands, amin=1e-7)


def audio_preprocessing(audio_file, mean, std):
    mels = get_log_melspectrogram(audio_file)
    if mels.shape[-1] != 3751:
        print()
    mels = (mels - mean[:, None]) / std[:, None]  # normalize
    return mels.astype('float32')


def save_mean_std_for_pretrain(train_names, mean_file, std_file):
    n = 0
    mean = np.zeros(cfg.N_MELS)
    var = np.zeros(cfg.N_MELS)
    for name in train_names:
        audio_f = os.path.join('data/ms/audio', name + '.wav')
        bands = get_log_melspectrogram(audio_f)
        length = bands.shape[1]

        if length > 0:
            n += length
            delta1 = bands - mean[:, None]
            mean += np.sum(delta1, axis=1) / n
            delta2 = bands - mean[:, None]
            var += np.sum(delta1 * delta2, axis=1)
    var /= (n - 1)
    np.save(mean_file, mean)
    np.save(std_file, var)


def save_mean_std(train_names, mean_file, std_file):
    n = 0
    mean = np.zeros(cfg.N_MELS)
    var = np.zeros(cfg.N_MELS)
    for name in train_names:
        audio_f = os.path.join(cfg.AUDIO_DIR, name + '.wav')
        bands = get_log_melspectrogram(audio_f)
        length = bands.shape[1]

        if length > 0:
            n += length
            delta1 = bands - mean[:, None]
            mean += np.sum(delta1, axis=1) / n
            delta2 = bands - mean[:, None]
            var += np.sum(delta1 * delta2, axis=1)
    var /= (n - 1)
    np.save(mean_file, mean)
    np.save(std_file, var)


def annotation_to_label_for_pretrain(filename, n_frame, stretching_rate=1):
    """Generate the label matrix of an audio sample based on its annotations."""

    def read_annotation(filename):
        events = []
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in spamreader:
                events.append(row)
        return events

    def time_to_frame(time):
        """Return the number of the frame corresponding to a timestamp."""
        n_frame = round(time / cfg.HOP_LENGTH * cfg.SAMPLING_RATE)
        return int(n_frame)

    events = read_annotation(filename)
    label = np.zeros((2, n_frame), dtype=int)

    if events == [["no-music"]]:
        label[0] = 1
    elif events == [["music"]]:
        label[1] = 1
    else:
        for event in events:
            f1 = time_to_frame(float(event[0]) * stretching_rate)
            f2 = time_to_frame(float(event[1]) * stretching_rate)
            if f2 < f1:
                raise ValueError("An error occured in the annotation file " + filename + ", f1 > f2.")

            if event[2] == 'no-music':
                label[0][f1:f2] = 1
            elif event[2] == 'music':
                label[1][f1:f2] = 1
            else:
                raise ValueError("An error occured in the annotation file " + filename + ", unknown type.")

    return label


def annotation_to_label(anno_file, n_frame, stretching_rate=1):
    """Generate the label matrix of an audio sample based on its annotations."""

    def read_annotation(filename):
        events = []
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in spamreader:
                events.append(row)
        return events

    def time_to_frame(time):
        """Return the number of the frame corresponding to a timestamp."""
        n_frame = round(time / cfg.HOP_LENGTH * cfg.SAMPLING_RATE)
        return int(n_frame)

    events = read_annotation(anno_file)
    label = np.zeros((3, n_frame), dtype=int)

    if events == [["no-music"]]:
        label[0] = 1
    elif events == [["fg-music"]]:
        label[1] = 1
    elif events == [["bg-music"]]:
        label[2] = 1
    else:
        for event in events:
            f1 = time_to_frame(float(event[0]) * stretching_rate)
            f2 = time_to_frame(float(event[1]) * stretching_rate)
            if f2 < f1:
                raise ValueError("An error occured in the annotation file " + anno_file + ", f1 > f2.")

            if event[2] == 'no-music':
                label[0][f1:f2] = 1
            elif event[2] == 'fg-music':
                label[1][f1:f2] = 1
            elif event[2] == 'bg-music':
                label[2][f1:f2] = 1
            else:
                raise ValueError("An error occured in the annotation file " + anno_file + ", unknown type.")

    return label


## Post-processing
def apply_threshold1(output, no_music_threshold=0.5, music_threshold=0.5):
    output[0] = np.where(output[0] >= no_music_threshold, 1, 0)
    output[1] = np.where(output[1] >= music_threshold, 1, 0)
    return output.astype(int)


def apply_threshold2(output, no_music_threshold=0.5, fg_music_threshold=0.5, bg_music_threshold=0.5):
    output[0] = np.where(output[0] >= no_music_threshold, 1, 0)
    output[1] = np.where(output[1] >= fg_music_threshold, 1, 0)
    output[2] = np.where(output[2] >= bg_music_threshold, 1, 0)
    return output.astype(int)


def smooth_output1(output, min_no_music=1.3, min_music=3.4, max_silence_no_music=0.4, max_silence_music=0.6):
    duration_frame = cfg.HOP_LENGTH / cfg.SAMPLING_RATE
    n_frame = output.shape[1]

    start_music = -1000
    start_no_music = -1000

    for i in range(n_frame):
        if output[0, i] == 1:
            if i - start_no_music > 1:
                if (i - start_no_music) * duration_frame <= max_silence_no_music:
                    output[0, start_no_music:i] = 1

            start_no_music = i

        if output[1, i] == 1:
            if i - start_music > 1:
                if (i - start_music) * duration_frame <= max_silence_music:
                    output[1, start_music:i] = 1

            start_music = i

    start_music = -1000
    start_no_music = -1000

    for i in range(n_frame):
        if i != n_frame - 1:
            if output[0, i] == 0:
                if i - start_no_music > 1:
                    if (i - start_no_music) * duration_frame <= min_no_music:
                        output[0, start_no_music:i] = 0

                start_no_music = i

            if output[1, i] == 0:
                if i - start_music > 1:
                    if (i - start_music) * duration_frame <= min_music:
                        output[1, start_music:i] = 0

                start_music = i
        else:
            if i - start_no_music > 1:
                if (i - start_no_music) * duration_frame <= min_no_music:
                    output[0, start_no_music:i + 1] = 0

            if i - start_music > 1:
                if (i - start_music) * duration_frame <= min_music:
                    output[1, start_music:i + 1] = 0

    return output


# TODO: 改为针对mrle
def smooth_output2(output, min_speech=1.3, min_music=3.4, max_silence_speech=0.4, max_silence_music=0.6):
    duration_frame = cfg.HOP_LENGTH / cfg.SAMPLING_RATE
    n_frame = output.shape[1]

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if output[0, i] == 1:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= max_silence_speech:
                    output[0, start_speech:i] = 1

            start_speech = i

        if output[1, i] == 1:
            if i - start_music > 1:
                if (i - start_music) * duration_frame <= max_silence_music:
                    output[1, start_music:i] = 1

            start_music = i

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if i != n_frame - 1:
            if output[0, i] == 0:
                if i - start_speech > 1:
                    if (i - start_speech) * duration_frame <= min_speech:
                        output[0, start_speech:i] = 0

                start_speech = i

            if output[1, i] == 0:
                if i - start_music > 1:
                    if (i - start_music) * duration_frame <= min_music:
                        output[1, start_music:i] = 0

                start_music = i
        else:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= min_speech:
                    output[0, start_speech:i + 1] = 0

            if i - start_music > 1:
                if (i - start_music) * duration_frame <= min_music:
                    output[1, start_music:i + 1] = 0

    return output


def label_to_annotation(label, task='mrle'):
    """Return the formatted annotations based on the label matrix."""

    def frame_to_time(n_frame):
        """Return the timestamp corresponding to a frame."""
        time = n_frame / cfg.SAMPLING_RATE * cfg.HOP_LENGTH
        return time

    events = []

    if task == 'mrle':
        t1_no_music = -1
        t1_fg_music = -1
        t1_bg_music = -1

        for i in range(label.shape[1]):  # for each time step
            if label[0][i] == 1 and t1_no_music == -1:
                t1_no_music = frame_to_time(i)
            elif label[0][i] == 0 and t1_no_music != -1:
                t2_no_music = frame_to_time(i)
                events.append([str(t1_no_music), str(t2_no_music), "no-music"])
                t1_no_music = -1

            if label[1][i] == 1 and t1_fg_music == -1:
                t1_fg_music = frame_to_time(i)
            elif label[1][i] == 0 and t1_fg_music != -1:
                t2_fg_music = frame_to_time(i)
                events.append([str(t1_fg_music), str(t2_fg_music), "fg-music"])
                t1_fg_music = -1

            if label[2][i] == 1 and t1_bg_music == -1:
                t1_bg_music = frame_to_time(i)
            elif label[2][i] == 0 and t1_bg_music != -1:
                t2_bg_music = frame_to_time(i)
                events.append([str(t1_bg_music), str(t2_bg_music), "bg-music"])
                t1_bg_music = -1

        if t1_no_music != -1:
            t2_no_music = frame_to_time(len(label[0]))
            events.append([str(t1_no_music), str(t2_no_music), "no-music"])

        if t1_fg_music != -1:
            t2_fg_music = frame_to_time(len(label[0]))
            events.append([str(t1_fg_music), str(t2_fg_music), "fg-music"])

        if t1_bg_music != -1:
            t2_bg_music = frame_to_time(len(label[0]))
            events.append([str(t1_bg_music), str(t2_bg_music), "bg-music"])

    elif task == 'mud':
        t1_no_music = -1
        t1_music = -1

        for i in range(len(label[0])):
            if label[0][i] == 1 and t1_no_music == -1:
                t1_no_music = frame_to_time(i)
            elif label[0][i] == 0 and t1_no_music != -1:
                t2_no_music = frame_to_time(i)
                events.append([str(t1_no_music), str(t2_no_music), "no-music"])
                t1_no_music = -1

            if label[1][i] == 1 and t1_music == -1:
                t1_music = frame_to_time(i)
            elif label[1][i] == 0 and t1_music != -1:
                t2_music = frame_to_time(i)
                events.append([str(t1_music), str(t2_music), "music"])
                t1_music = -1

        if t1_no_music != -1:
            t2_no_music = frame_to_time(len(label[0]))
            events.append([str(t1_no_music), str(t2_no_music), "no-music"])

        if t1_music != -1:
            t2_music = frame_to_time(len(label[0]))
            events.append([str(t1_music), str(t2_music), "music"])

    else:
        raise ValueError("Only takes `mud` and `mrle`. `{}` is invalid.".format(task))
    return events


def save_annotation2(events, filename, dst=None):
    r"""Save the annotation of an audio file based on the event list.
    The event list has to be formatted this way:
    [
    [t_start, t_end, label],
    [t_start, t_end, label],
    ...
    ]
    or
    [[no-music]], [[fg-music]], [[bg-music]]
    """
    if dst is None:
        path = filename
    else:
        path = os.path.join(dst, filename)

    with open(path, "w") as f:
        if events == [["no-music"]]:
            f.write("no-music")
        elif events == [["fg-music"]]:
            f.write("fg-music")
        elif events == [["bg-music"]]:
            f.write("bg-music")
        else:
            events = sorted(events, key=lambda x: float(x[0]))
            for event in events:
                f.write(str(event[0]) + '\t' + str(event[1]) + '\t' + event[2] + '\n')


def mrle_label_to_mud_label(mrle_label):
    mud_label = np.empty((2, mrle_label.shape[1]), dtype=int)
    mud_label[0] = mrle_label[0]
    mud_label[1] = mrle_label[1] | mrle_label[2]
    return mud_label


## Create Dataset and DataLoader
class MS(data.Dataset):
    def __init__(self, name_list, mean_file, std_file):
        self.data = name_list
        self.train_mean = np.load(mean_file)
        self.train_std = np.load(std_file)

    def __getitem__(self, idx):
        base_name = self.data[idx]
        audio_file = os.path.join('data/ms/audio', base_name + '.wav')
        anno_file = os.path.join('data/ms/annotations', base_name + '.tsv')

        x_data = audio_preprocessing(audio_file, self.train_mean, self.train_std)
        x = torch.from_numpy(x_data)  # [n_mels, T]

        mud_y_data = annotation_to_label_for_pretrain(anno_file, x.shape[1])

        mud_y = torch.from_numpy(mud_y_data)  # [2, T]

        return x.T, mud_y.float().T

    def __len__(self):
        return len(self.data)


class OpenBMAT(data.Dataset):
    def __init__(self, name_list, mean_file, std_file):
        self.data = name_list
        self.train_mean = np.load(mean_file)
        self.train_std = np.load(std_file)

    def __getitem__(self, idx):
        base_name = self.data[idx]
        audio_file = os.path.join(cfg.AUDIO_DIR, base_name + '.wav')
        anno_file = os.path.join(cfg.ANNO_DIR, base_name + '.tsv')

        x_data = audio_preprocessing(audio_file, self.train_mean, self.train_std)
        x = torch.from_numpy(x_data)  # [n_mels, T]

        mrle_y_data = annotation_to_label(anno_file, x.shape[1])
        mud_y_data = mrle_label_to_mud_label(mrle_y_data)

        mud_y = torch.from_numpy(mud_y_data)  # [2, T]
        mrle_y = torch.from_numpy(mrle_y_data)  # [3, T]

        return x.T, mud_y.float().T, mrle_y.float().T

    def __len__(self):
        return len(self.data)


# ---------- Model ----------
def build_model(model_cfg):
    model = None

    if model_cfg["type"] == "lstm":
        from models.lstm import BiLSTM
        model = BiLSTM(cfg.N_MELS, model_cfg["hidden_size"], model_cfg["num_layers"],
                       model_cfg["dropout"], model_cfg["bidirectional"])

    if model_cfg["type"] == "gru":
        from models.gru import BiGRU
        model = BiGRU(cfg.N_MELS, model_cfg["hidden_size"], model_cfg["num_layers"],
                      model_cfg["dropout"], model_cfg["bidirectional"])

    if model_cfg["type"] == "cldnn":
        from models.cldnn import CLDNN
        model = CLDNN(model_cfg["n_filters"], model_cfg["kernel_sizes"], model_cfg["strides"],
                      model_cfg["dropout"], model_cfg["rnn_hid_size"], model_cfg["rnn_n_layers"],
                      model_cfg["bidirectional"])

    if model_cfg["type"] == "hedln":
        from models.hedln import HEDLN
        model = HEDLN(cfg.N_MELS, model_cfg["hidden_size"], model_cfg["num_layers"],
                      model_cfg["dropout"], model_cfg["bidirectional"], cfg.NUM_CLASSES1, cfg.NUM_CLASSES2)

    if model_cfg["type"] == "hedln_r":
        from models.hedln_r import HEDLN_R
        model = HEDLN_R(cfg.N_MELS, model_cfg["hidden_size"], model_cfg["num_layers"],
                        model_cfg["dropout"], model_cfg["bidirectional"], cfg.NUM_CLASSES1, cfg.NUM_CLASSES2)

    if model_cfg["type"] == "hedln_cr":
        from models.hedln_cr import HEDLN_CR
        model = HEDLN_CR(model_cfg["n_filters"], model_cfg["kernel_sizes"], model_cfg["strides"],
                         model_cfg["dropout"], model_cfg["rnn_hid_size"], model_cfg["rnn_n_layers"],
                         model_cfg["bidirectional"])

    return model


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# ---------- Evaluation ----------
def eval(ground_truth_events, predicted_events, segment_length, event_tolerance, offset=False):
    r"""
        Evaluate the output of the network.
        ground_truth_events and predicted_events can either be:
            - a list of path to containing the events (one path for one audio)
            - a path containing the events (for one audio)
            - a list of list containing the events (a list of the events of different audio)
            - a list containing the events (the events of one audio)
        The segment_length and the event_tolerance are giving in s.
        If one of those two parameters is set to None, this evaluation will be skipped.
    """
    data = []
    all_data = dcase_util.containers.MetaDataContainer()

    if type(ground_truth_events) == str and type(predicted_events) == str:
        reference_event_list = sed_eval.io.load_event_list(ground_truth_events)
        estimated_event_list = sed_eval.io.load_event_list(predicted_events)

        data.append({'reference_event_list': reference_event_list,
                     'estimated_event_list': estimated_event_list})

        all_data += reference_event_list

    elif type(ground_truth_events) == list and type(predicted_events) == list:
        if len(ground_truth_events) != len(predicted_events):
            raise ValueError("The two lists must have the same size.")
        if all(isinstance(n, str) for n in ground_truth_events) and all(isinstance(n, str) for n in predicted_events):

            for i in range(len(ground_truth_events)):
                reference_event_list = sed_eval.io.load_event_list(ground_truth_events[i])
                estimated_event_list = sed_eval.io.load_event_list(predicted_events[i])

                data.append({'reference_event_list': reference_event_list,
                             'estimated_event_list': estimated_event_list})

                all_data += reference_event_list

        if all(isinstance(n, list) for n in ground_truth_events) and all(isinstance(n, list) for n in predicted_events):
            if all(isinstance(x, list) for b in ground_truth_events for x in b) and all(
                    isinstance(x, list) for b in predicted_events for x in b):
                for gt, p in zip(ground_truth_events, predicted_events):
                    formatted_gt_events = []
                    formatted_p_events = []

                    for event in gt:
                        formatted_gt_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                    for event in p:
                        formatted_p_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                    formatted_p_events = dcase_util.containers.MetaDataContainer(formatted_p_events)
                    formatted_gt_events = dcase_util.containers.MetaDataContainer(formatted_gt_events)

                    data.append({'reference_event_list': formatted_gt_events,
                                 'estimated_event_list': formatted_p_events})

                    all_data += formatted_gt_events
            else:
                formatted_gt_events = []
                formatted_p_events = []

                for event in ground_truth_events:
                    formatted_gt_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                for event in predicted_events:
                    formatted_p_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                formatted_p_events = dcase_util.containers.MetaDataContainer(formatted_p_events)
                formatted_gt_events = dcase_util.containers.MetaDataContainer(formatted_gt_events)

                data.append({'reference_event_list': formatted_gt_events,
                             'estimated_event_list': formatted_p_events})

                #                 all_data += reference_event_list   # 这行有问题，reference_event_list未定义，jbj改为下面这行
                all_data += formatted_gt_events
    else:
        raise ValueError("Incorrect input format.")

    event_labels = all_data.unique_event_labels

    if not (segment_length is None):
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=event_labels,
            time_resolution=segment_length
        )

    if not (event_tolerance is None):
        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=event_labels,
            t_collar=event_tolerance,
            percentage_of_length=0.,
            evaluate_onset=True,
            evaluate_offset=offset
        )

    for file_pair in data:
        if not (segment_length is None):
            segment_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )
        if not (event_tolerance is None):
            event_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )

    if not (event_tolerance is None) and not (segment_length is None):
        print(segment_based_metrics)
        print(event_based_metrics)
        return segment_based_metrics, event_based_metrics
    elif event_tolerance is None and not (segment_length is None):
        print(segment_based_metrics)
        return segment_based_metrics
    elif not (event_tolerance is None) and segment_length is None:
        print(event_based_metrics)
        return event_based_metrics
    else:
        return
