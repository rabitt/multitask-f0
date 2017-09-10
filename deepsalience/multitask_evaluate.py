"""Evaluation functions
"""
from __future__ import print_function
import numpy as np
import scipy
import os
import pandas as pd
import glob
import json
import mir_eval

import compute_training_data as C
import multitask_core as MC

TEST_DIR = '/scratch/rmb456/multif0_ismir2017/multitask_test_data/'
VALIDATION_DIR = os.path.join(
    "/scratch/rmb456/multif0_ismir2017",
    "multitask_test_data/validation_data/"
)


def get_single_test_prediction(model, npy_file=None, audio_file=None, max_frames=None,
                               add_frequency=False, n_harms=5):
    """Generate output from a model given an input numpy file
    """
    if npy_file is not None:
        input_hcqt = np.load(npy_file)
    elif audio_file is not None:
        input_hcqt = (C.compute_hcqt(audio_file)).astype(np.float32)
    else:
        raise ValueError("one of npy_file or audio_file must be specified")

    x = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]

    if max_frames is not None:
        x = x[:, :, :max_frames, :]

    n_t = x.shape[2]
    n_f = x.shape[1]
    n_slices = 1000
    t_slices = list(np.arange(0, n_t, n_slices))
    model_output = model.output
    if isinstance(model_output, list):
        output_list = [[] for i in range(len(model_output))]
        predicted_output = [[] for i in range(len(model_output))]
    else:
        output_list = [[]]
        predicted_output = [[]]

    for t in t_slices:

        x_slice = x[:, :, t:t+n_slices, :]

        if add_frequency:
            x_freq = MC.get_freq_feature(n_f, x_slice.shape[2], augment=False)
            x_in = {'input': x_slice, 'freq_map': x_freq}
        else:
            x_in = x_slice[:, :, :, :n_harms]

        prediction = model.predict(x_in)

        if isinstance(prediction, list):
            for i, pred in enumerate(prediction):
                output_list[i].append(pred[0, :, :])
        else:
            output_list[0].append(prediction[0, :, :])

    for i in range(len(output_list)):
        predicted_output[i] = np.hstack(output_list[i])

    return predicted_output, x


def pitch_activations_to_singlef0(pitch_activation_mat, thresh, use_neg=True):
    max_idx = np.argmax(pitch_activation_mat, axis=0)
    est_times = C.get_time_grid(pitch_activation_mat.shape[1])
    freq_grid = C.get_freq_grid()
    est_freqs = []
    for i, f in enumerate(max_idx):
        if pitch_activation_mat[f, i] < thresh:
            if use_neg:
                est_freqs.append(-1.0*freq_grid[f])
            else:
                est_freqs.append(0.0)
        else:
            est_freqs.append(freq_grid[f])
    est_freqs = np.array(est_freqs)
    return est_times, est_freqs


def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert a pitch activation map to multif0 by thresholding peak values
    at thresh
    """
    freqs = C.get_freq_grid()
    times = C.get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(times))]
    est_amps = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freqs[f])
        est_amps[t].append(pitch_activation_mat[f, t])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return times, est_freqs, est_amps


def get_validation_files(task):
    task_path = os.path.join(VALIDATION_DIR, task)
    npy_files = glob.glob(os.path.join(task_path, "*.npy"))
    validation_files = []
    for npy_file in npy_files:
        trackid = '_'.join(
            os.path.basename(npy_file).split('.')[0].split('_')[:2])
        txt_file = glob.glob(os.path.join(task_path, "{}*.txt".format(trackid)))
        if len(txt_file) > 0:
            txt_file = txt_file[0]
            validation_files.append([npy_file, txt_file])
        else:
            print('coudnt find text path')

    return validation_files


def time_freq_to_ragged_time_series(times, freqs, hop):
    max_time = np.max(times)
    t_uniform = np.arange(0, max_time, hop)
    time_idx = np.digitize(times, t_uniform) - 1

    freq_list = [[] for _ in t_uniform]
    for i, f in zip(time_idx, freqs):
        if f > 0:
            freq_list[i].append(f)

    freq_arrays = [np.array(lst) for lst in freq_list]
    return t_uniform, freq_arrays


def time_freq_fill_zeros(times, freqs, hop):
    max_time = np.max(times)
    t_uniform = np.arange(0, max_time, hop)
    time_idx = np.digitize(times, t_uniform) - 1
    freq_array = np.zeros(t_uniform.shape)
    for i, f in zip(time_idx, freqs):
        freq_array[i] = f

    return t_uniform, freq_array


def get_best_thresh_multif0(model, task_indices, task='multif0', add_frequency=False,
                            n_harms=5):
    """Use validation set to get the best threshold value
    """
    # get validation files
    validation_files = get_validation_files('multif0')

    thresh_vals = np.arange(0.1, 1.0, 0.1)
    thresh_scores = {t: [] for t in thresh_vals}
    n_validation = len(validation_files)
    for i, (npy_file, target_file) in enumerate(validation_files):
        print("    > {} / {}".format(i + 1, n_validation))

        # generate prediction on numpy file
        predicted_outputs, _ = \
            get_single_test_prediction(
                model, npy_file=npy_file, add_frequency=add_frequency,
                n_harms=n_harms)
        predicted_output = predicted_outputs[task_indices[task]]

        # load ground truth labels
        temp_times, temp_freqs = mir_eval.io.load_time_series(target_file)
        ref_times, ref_freqs = time_freq_to_ragged_time_series(
            temp_times, temp_freqs, hop=256./44100.)

        for thresh in thresh_vals:
            # get multif0 output from prediction
            est_times, est_freqs, _ = \
                pitch_activations_to_mf0(predicted_output, thresh)

            # get multif0 metrics and append
            scores = mir_eval.multipitch.evaluate(
                ref_times, ref_freqs, est_times, est_freqs)
            thresh_scores[thresh].append(scores['Accuracy'])

    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    print("Best Threshold is {}".format(best_thresh))
    print("Best validation accuracy is {}".format(np.max(avg_thresh)))
    print("Validation accuracy at 0.5 is {}".format(
        np.mean(thresh_scores[0.5])))

    return best_thresh


def get_best_thresh_singlef0(model, task, task_indices, add_frequency=False,
                             n_harms=5):
    thresh_vals = np.arange(0, 1, 0.1)
    mel_accuracy = {v: [] for v in thresh_vals}

    validation_files = get_validation_files(task)

    n_validation = len(validation_files)
    for i, (npy_file, target_file) in enumerate(validation_files):
        print("    > {} / {}".format(i + 1, n_validation))
        # generate prediction on numpy file
        predicted_outputs, _ = \
            get_single_test_prediction(
                model, npy_file=npy_file, add_frequency=add_frequency,
                n_harms=n_harms)

        predicted_output = predicted_outputs[task_indices[task]]

        temp_times, temp_freqs = mir_eval.io.load_time_series(target_file)
        ref_times, ref_freqs = time_freq_fill_zeros(
            temp_times, temp_freqs, hop=256./44100.)

        for thresh in thresh_vals:
            est_times, est_freqs = pitch_activations_to_singlef0(
                predicted_output, thresh)
            mel_scores = mir_eval.melody.evaluate(
                ref_times, ref_freqs, est_times, est_freqs)
            mel_accuracy[thresh].append(mel_scores['Overall Accuracy'])

    avg_thresh = [np.mean(mel_accuracy[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    print("Best Threshold is {}".format(best_thresh))
    print("Best validation accuracy is {}".format(np.max(avg_thresh)))
    print("Validation accuracy at 0.5 is {}".format(np.mean(mel_accuracy[0.5])))

    return best_thresh


def get_test_files(test_set_name):
    if test_set_name in ['bach10', 'maps', 'medleydb_multif0', 'su']:
        test_dir = os.path.join(TEST_DIR, test_set_name)
        extension = 'txt'
    elif test_set_name in ['ikala', 'medleydb_melody', 'orchset']:
        test_dir = os.path.join(TEST_DIR, test_set_name)
        extension = 'csv'
    elif test_set_name == 'weimar_jazz_bass':
        test_dir = os.path.join(TEST_DIR, 'weimar_jazz', 'bass')
        extension = 'csv'
    elif test_set_name == 'weimar_jazz_melody':
        test_dir = os.path.join(TEST_DIR, 'weimar_jazz', 'melody')
        extension = 'csv'

    npy_files = glob.glob(os.path.join(test_dir, "*.npy"))
    test_files = []
    for npy_file in npy_files:
        trackid = '_'.join(
            os.path.basename(npy_file).split('.')[0].split('_')[:2])
        txt_file = glob.glob(
            os.path.join(test_dir, "{}*.{}".format(trackid, extension)))
        if len(txt_file) > 0:
            txt_file = txt_file[0]
            test_files.append([npy_file, txt_file])
        else:
            print('coudnt find text path')

    return test_files


def score_singlef0_on_test_set(model, test_set, task_indices, thresh=0.5,
                               add_frequency=False, n_harms=5):
    test_files = get_test_files(test_set)
    print("Scoring on {}".format(test_set))


    if test_set == 'orchset':
        delimiter = '\t'
    else:
        delimiter = ','

    if test_set in ['ikala']:
        task_idx = task_indices['vocal']
    elif test_set in ['medleydb_melody', 'orchset', 'weimar_jazz_melody']:
        task_idx = task_indices['melody']
    elif test_set in ['weimar_jazz_bass']:
        task_idx = task_indices['bass']
    else:
        raise ValueError('invalid test set')

    all_scores = []
    n_test_files = len(test_files)
    for i, (npy_file, target_file) in enumerate(test_files):
        print("    > {} / {}".format(i + 1, n_test_files))
        # generate prediction on numpy file
        predicted_outputs, _ = \
            get_single_test_prediction(
                model, npy_file=npy_file, add_frequency=add_frequency,
                n_harms=n_harms)
        predicted_output = predicted_outputs[task_idx]

        file_keys = os.path.basename(npy_file).split('.')[0]

        ref_times, ref_freqs = mir_eval.io.load_time_series(
            target_file, delimiter=delimiter)
        est_times, est_freqs = pitch_activations_to_singlef0(
            predicted_output, thresh)
        mel_scores = mir_eval.melody.evaluate(
            ref_times, ref_freqs, est_times, est_freqs)
        if isinstance(file_keys, list):
            mel_scores['track'] = '_'.join(file_keys)
        else:
            mel_scores['track'] = file_keys
        all_scores.append(mel_scores)

    df = pd.DataFrame(all_scores)
    print(df.describe())
    return df


def score_multif0_on_test_set(model, test_set, task_indices, thresh=0.5, add_frequency=False,
                              n_harms=5):
    """score a model on all files in a named test set
    """

    # get files for this test set
    test_files = get_test_files(test_set)
    print("    > Scoring on {}...".format(test_set))
    task_idx = task_indices['multif0']

    all_scores = []
    n_test_files = len(test_files)
    for i, (npy_file, target_file) in enumerate(test_files):

        print("    > {} / {}".format(i + 1, n_test_files))
        file_keys = os.path.basename(npy_file).split('.')[0]

        # generate prediction on numpy file
        predicted_outputs, _ = \
            get_single_test_prediction(
                model, npy_file=npy_file, add_frequency=add_frequency,
                n_harms=n_harms)

        predicted_output = predicted_outputs[task_idx]
        # get multif0 output from prediction
        est_times, est_freqs, _ = pitch_activations_to_mf0(
            predicted_output, thresh
        )

        # load ground truth labels
        ref_times, ref_freqs = \
            mir_eval.io.load_ragged_time_series(target_file)

        # get multif0 metrics and append
        scores = mir_eval.multipitch.evaluate(
            ref_times, ref_freqs, est_times, est_freqs)
        scores['track'] = '_'.join(file_keys)
        all_scores.append(scores)

    df = pd.DataFrame(all_scores)
    print(df.describe())
    return df


def evaluate_model(model, tasks, task_indices, add_frequency=False,
                   n_harms=5):

    thresholds = {}
    scores = {}

    print("Computing Multif0 Metrics...")
    if 'multif0' in tasks:
        print("    > Getting best threshold...")
        best_thresh_mf0 = get_best_thresh_multif0(
            model, task_indices, add_frequency=add_frequency, n_harms=n_harms)
        thresholds['multif0'] = best_thresh_mf0

        df_bach10 = score_multif0_on_test_set(
            model, 'bach10', task_indices, best_thresh_mf0,
            add_frequency=add_frequency, n_harms=n_harms)
        df_maps = score_multif0_on_test_set(
            model, 'maps', task_indices, best_thresh_mf0,
            add_frequency=add_frequency, n_harms=n_harms)
        df_mdb_mf0 = score_multif0_on_test_set(
            model, 'medleydb_multif0', task_indices, best_thresh_mf0,
            add_frequency=add_frequency, n_harms=n_harms)
        df_su = score_multif0_on_test_set(
            model, 'su', task_indices, best_thresh_mf0,
            add_frequency=add_frequency, n_harms=n_harms)

        scores['bach10'] = df_bach10
        scores['maps'] = df_maps
        scores['mdb_mf0'] = df_mdb_mf0
        scores['su'] = df_su

    if 'melody' in tasks:
        print("    > Getting best threshold...")
        best_thresh_mel = get_best_thresh_singlef0(
            model, 'melody', task_indices, add_frequency=add_frequency,
            n_harms=n_harms)
        thresholds['melody'] = best_thresh_mel

        df_mdb_mel = score_singlef0_on_test_set(
            model, 'medleydb_melody', task_indices, best_thresh_mel,
            add_frequency=add_frequency, n_harms=n_harms)
        df_orchset = score_singlef0_on_test_set(
            model, 'orchset', task_indices, best_thresh_mel,
            add_frequency=add_frequency, n_harms=n_harms)
        df_wj_mel = score_singlef0_on_test_set(
            model, 'weimar_jazz_melody', task_indices, best_thresh_mel,
            add_frequency=add_frequency, n_harms=n_harms)

        scores['mdb_mel'] = df_mdb_mel
        scores['orchset'] = df_orchset
        scores['wj_mel'] = df_wj_mel

    if 'bass' in tasks:
        print("    > Getting best threshold...")
        best_thresh_bass = get_best_thresh_singlef0(
            model, 'bass', task_indices, add_frequency=add_frequency,
            n_harms=n_harms)
        thresholds['bass'] = best_thresh_bass

        df_wj_bass = score_singlef0_on_test_set(
            model, 'weimar_jazz_bass', task_indices, best_thresh_bass,
            add_frequency=add_frequency, n_harms=n_harms)

        scores['wj_bass'] = df_wj_bass

    if 'vocal' in tasks:
        print("    > Getting best threshold...")
        best_thresh_vocal = get_best_thresh_singlef0(
            model, 'vocal', task_indices, add_frequency=add_frequency,
            n_harms=n_harms)

        thresholds['vocal'] = best_thresh_vocal

        df_ikala = score_singlef0_on_test_set(
            model, 'ikala', task_indices, best_thresh_vocal,
            add_frequency=add_frequency, n_harms=n_harms)

        scores['ikala'] = df_ikala

    return thresholds, scores


def save_eval(output_path, thresholds, scores):
    save_path = os.path.join(output_path, 'evaluation')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print("Saving Thresholds...")
    threshold_path = os.path.join(save_path, 'thresholds.json')
    with open(threshold_path, 'w') as fhandle:
        json.dump(thresholds, fhandle)

    score_summary = {}
    for key in scores.keys():
        print("Saving scores for {}...".format(key))
        df = scores[key]
        score_path = os.path.join(save_path, "{}_scores.csv".format(key))
        df.to_csv(score_path)

        headers = list(df.columns.values)
        if 'Accuracy' in headers:
            score_summary[key] = df['Accuracy'].mean()
        elif 'Overall Accuracy' in headers:
            score_summary[key] = df['Overall Accuracy'].mean()
        else:
            print("couldn't find correct header to create summary: ")
            print(headers)

    print("Saving Score Summary...")
    score_summary_path = os.path.join(save_path, 'score_summary.json')
    with open(score_summary_path, 'w') as fhandle:
        json.dump(score_summary, fhandle)

    print("Done!")
