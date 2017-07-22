from __future__ import print_function

import glob
import json
import librosa
import medleydb as mdb
import mir_eval
import numpy as np
import os
import pandas as pd
import pescador

import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda, Permute
from keras.layers.merge import Concatenate, Multiply
from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import compute_training_data as C
# import experiment_datasets
# import evaluate
# import core

DATA_TYPES = ['XA', 'XB', 'XC', 'XD']
TASKS = ['multif0', 'melody', 'bass', 'vocal']
JSON_PATH = "/scratch/rmb456/multif0_ismir2017/multitask_data/XY_pairs"
DATA_SPLITS_PATH = "/scratch/rmb456/multif0/outputs/data_splits.json"


def load_data_splits():
    with open(DATA_SPLITS_PATH, 'r') as fhandle:
        data_splits = json.load(fhandle)
    return data_splits


def get_grouped_data(json_path, mtrack_list):

    json_fpaths = glob.glob(os.path.join(json_path, '*.json'))
    typed_data = {'XA': [], 'XB': [], 'XC': [], 'XD': []}

    for fpath in json_fpaths:

        track_id = '_'.join(
            os.path.basename(fpath).split('.')[0].split('_')[:2])
        if track_id not in mtrack_list:
            continue

        with open(fpath, 'r') as fhandle:
            dat = json.load(fhandle)

            for key in dat.keys():
                bname = os.path.basename(key).split('.')[0].split('_')
                if len(bname) == 4:
                    typed_data['XA'].append([key, dat[key]])
                elif len(bname) == 6:
                    if bname[4] == 'resynth':
                        typed_data['XB'].append([key, dat[key]])
                    elif bname[4] == 'noguitar':
                        typed_data['XC'].append([key, dat[key]])
                    elif bname[4] == 'nosynth':
                        typed_data['XD'].append([key, dat[key]])
                    else:
                        raise ValueError(
                            "bname[4] = {} not recognized".format(bname[4]))
                else:
                    raise ValueError(
                        "len(bname) = {} not recognized".format(len(bname)))

    return typed_data


def grab_patch_output(f, t, n_f, n_t, y_data):
    """Get a time-frequency patch from an output file
    """
    return y_data[f: f + n_f, t: t + n_t][np.newaxis, :, :]


def grab_patch_input(f, t, n_f, n_t, x_data):
    """Get a time-frequency patch from an input file
    """
    return np.transpose(
        x_data[:, f: f + n_f, t: t + n_t], (1, 2, 0)
    )[np.newaxis, :, :, :]


def grab_empty_output(n_f, n_t):
    return np.zeros((1, n_f, n_t))


def multitask_patch_generator(fpath_in, dict_out, tasks, n_samples=20,
                              input_patch_size=(360, 50)):
    """Generator that yields an infinite number of patches
       for a single input, output pair
    """
    data_in = np.load(fpath_in)
    data_out = {}
    for task in dict_out.keys():
        data_out[task] = np.load(dict_out[task])

    _, _, n_times = data_in.shape
    n_f, n_t = input_patch_size

    t_vals = np.arange(0, n_times - n_t)
    np.random.shuffle(t_vals)

    for t in t_vals[:n_samples]:
        f = 0
        t = np.random.randint(0, n_times - n_t)

        x = grab_patch_input(
            f, t, n_f, n_t, data_in
        )
        y = {}
        w = {}
        for task in tasks:
            if task in data_out.keys():
                y_task = grab_patch_output(
                    f, t, n_f, n_t, data_out[task]
                )
                w[task] = 1.0
            else:
                y_task = grab_empty_output(n_f, n_t)
                w[task] = 0.0

            y[task] = y_task
            
        yield dict(X=x, Y=y, W=w)

        
def get_task_pairs(data_list, task):
    """Get a list of [input path, output dictionary] pairs where each
    element of the list has at least `task` as a key in the output dictionary.
    """
    task_pairs = []
    for input_file, task_dict in data_list:
        if task in task_dict.keys():
            task_pairs.append([input_file, task_dict])
    return task_pairs


def get_all_task_pairs(typed_data):
    task_pairs = {}
    for data_type in typed_data.keys():
        data_list = typed_data[data_type]
        multif0_pairs = get_task_pairs(data_list, 'multif0')
        melody_pairs = get_task_pairs(data_list, 'melody')
        bass_pairs = get_task_pairs(data_list, 'bass')
        vocal_pairs = get_task_pairs(data_list, 'vocal')
        task_pairs[data_type] = {
            'multif0': multif0_pairs,
            'melody': melody_pairs,
            'bass': bass_pairs,
            'vocal': vocal_pairs
        }
    return task_pairs


def multitask_batch_generator(task_generators, tasks):

    iterators = {}
    for task in tasks:
        iterators[task] = task_generators[task].tuples('X', 'Y', 'W')

    while True:
        next_samples = [next(iterators[task]) for task in tasks]
        X = np.concatenate([samp[0] for samp in next_samples])
        Y = {}
        W = {}
        for task in tasks:
            Y[task] = np.concatenate([samp[1][task] for samp in next_samples])
            W[task] = np.concatenate([samp[2][task] for samp in next_samples])

        yield (X, Y, W)


def bkld(y_true, y_pred):
    """Brian's KL Divergence implementation
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)


def soft_binary_accuracy(y_true, y_pred):
    """Binary accuracy that works when inputs are probabilities
    """
    return K.mean(K.mean(
        K.equal(K.round(y_true), K.round(y_pred)), axis=-1), axis=-1)


def multitask_generator(mtrack_list, json_path=JSON_PATH, data_types=DATA_TYPES,
                        tasks=TASKS, mux_weights=None):

    typed_data = get_grouped_data(json_path, mtrack_list)
    task_pairs = get_all_task_pairs(typed_data)
    
    # make a streamer for each data type and task
    data_streamers = {}
    for data_type in data_types:
        if data_type == 'XA':
            data_streamers[data_type] = {
                'melody': [], 'bass': [], 'vocal': []
            }
        else:
            data_streamers[data_type] = {
                'multif0': [], 'melody': [], 'bass': [], 'vocal': []
            }

        for task in data_streamers[data_type].keys():
            for pair in task_pairs[data_type][task]:
                data_streamers[data_type][task].append(
                    pescador.Streamer(multitask_patch_generator,
                                      tasks, pair[0], pair[1])
                )

    # for each data type make a mux
    n_active = 10
    data_muxes = {}
    for data_type in data_types:
        data_muxes[data_type] = {}
        for task in data_streamers[data_type].keys():
            data_muxes[data_type][task] = pescador.Mux(
                data_streamers[data_type][task], n_active,
                with_replacement=True, lam=250, random_state=42
            )

    # for each task make a mux that samples from the data muxes
    task_streams = {}
    for task in tasks:
        task_streams[task] = []
        for data_type in data_types:
            if task in data_muxes[data_type].keys():
                task_streams[task].append(data_muxes[data_type][task])

    if mux_weights is None:
        for task in tasks:
            n_data_types = len(task_streams[task])
            mux_weights[task] = np.ones((n_data_types, )) / float(n_data_types)


    task_muxes = {}
    for task in tasks:
        task_muxes[task] = pescador.Mux(
            task_streams[task], 1, pool_weights=mux_weights[task]
        )

    batch_gen = multitask_batch_generator(task_muxes, tasks)

    for batch in batch_gen:
        yield batch


def history_plot(history, tasks, save_path):
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 1, 1)
    plt.plot(history.history['loss'])
    for task in tasks:
        plt.plot(history.history['{}_loss'.format(task)])
    plt.title('Training Loss')
    plt.ylabel('Training Loss')
    plt.xlabel('epoch')
    plt.legend(['overall'] + tasks, loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(history.history['val_loss'])
    for task in tasks:
        plt.plot(history.history['val_{}_loss'.format(task)])
    plt.title('Validation Loss')
    plt.ylabel('Validation Loss')
    plt.xlabel('epoch')
    plt.legend(['overall'] + tasks, loc='upper left')

    plt.subplot(3, 1, 3)
    for task in tasks:
        plt.plot(history.history['val_{}_soft_binary_accuracy'.format(task)])
    plt.title('soft_binary_accuracy')
    plt.ylabel('soft_binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(tasks, loc='upper left')

    plt.savefig(save_path, format='pdf')
