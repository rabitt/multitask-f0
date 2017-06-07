"""Script to compute multitask X Y pairs
"""
from __future__ import print_function

import argparse
import compute_training_data as C
import glob
from joblib import Parallel, delayed
import json
import mir_eval
import numpy as np
import os


def compute_representations(audio_path, annotation_path, save_dir):

    X_bname = os.path.basename(audio_path).split('.')[0]
    X_save_path = os.path.join(save_dir, "{}_input.npy".format(X_bname))

    Y_bname = os.path.basename(annotation_path).split('.')[0]
    Y_save_path = os.path.join(save_dir, "{}_output.npy".format(Y_bname))
    
    if os.path.exists(X_save_path) and os.path.exists(Y_save_path):
        return None, None, X_save_path, Y_save_path

    if not os.path.exists(X_save_path):
        print("    > computing CQT for {}".format(
            os.path.basename(audio_path)))
        hcqt = C.compute_hcqt(audio_path)
    else:
        print("    > using precomputed CQT for {}".format(
            os.path.basename(audio_path)))
        hcqt = np.load(X_save_path, mmap_mode='r')

    freq_grid = C.get_freq_grid()
    time_grid = C.get_time_grid(len(hcqt[0][0]))

    if os.path.exists(annotation_path):
        annot_times, annot_freqs = mir_eval.io.load_time_series(annotation_path)
    else:
        annot_times = []
        annot_freqs = []

    annot_target = C.create_annotation_target(
        freq_grid, time_grid, annot_times, annot_freqs
    )

    return hcqt, annot_target, X_save_path, Y_save_path


def save_data(X, Y, X_save_path, Y_save_path):
    if X is not None and not os.path.exists(X_save_path):
        np.save(X_save_path, X.astype(np.float32))
        print("    Saved data to {}".format(X_save_path))

    if Y is not None and not os.path.exists(Y_save_path):
        np.save(Y_save_path, Y.astype(np.float32))
        print("    Saved data to {}".format(Y_save_path))


def get_XY_pairs(audio_path, annotation_path, save_dir): 
    X, Y, X_path, Y_path = compute_representations(
        audio_path, annotation_path, save_dir
    )
    save_data(X, Y, X_path, Y_path)
    return X_path, Y_path


def get_missing_annotation_path(audio_path, data_path, annotation_key):
    fparts = os.path.basename(audio_path).split('.')[0].split('_')
    track_id = '_'.join(fparts[:2])
    if len(fparts) == 3:
        annotation_bname = "{}_MIX_{}.txt".format(track_id, annotation_key)
    elif len(fparts) == 5:
        if annotation_key == 'multif0':
            if fparts[4] == 'resynth':
                annotation_bname = "{}_multif0_annotation.txt".format(track_id)
            elif fparts[4] == 'noguitar':
                annotation_bname = "{}_multif0_noguitar_annotation.txt".format(
                    track_id)
            elif fparts[4] == 'nosynth':
                annotation_bname = "{}_multif0_nosynth_annotation.txt".format(
                    track_id)
            else:
                raise ValueError("fparts[4] = {}".format(fparts[4]))
        else:
            annotation_bname = "{}_{}_annotation.txt".format(
                track_id, annotation_key)
    else:
        raise ValueError("fparts = {}".format(fparts))

    return os.path.join(data_path, annotation_bname)


def compute_labels_for_dict(json_file, data_path, save_dir):

    track_id = '_'.join(
        os.path.basename(json_file).split('.')[0].split('_')[:2])
    label_save_path = os.path.join(
        save_dir, '{}_XY_pairs.json'.format(track_id))
    if os.path.exists(label_save_path):
        print("{} already done!".format(track_id))
        return True

    with open(json_file, 'r') as fhandle:
        annot_dict = json.load(fhandle)

    label_dict = {}
    for audio_path in annot_dict.keys():
        for key, annot_path in annot_dict[audio_path].items():
            if annot_path is None:
                annot_path = get_missing_annotation_path(
                    audio_path, data_path, key)
            
            X_path, Y_path = get_XY_pairs(audio_path, annot_path, save_dir)
            if X_path not in label_dict.keys():
                label_dict[X_path] = {}

            label_dict[X_path][key] = Y_path

    with open(label_save_path, 'w') as fhandle:
        json.dump(label_dict, fhandle, indent=2)

    print("{} done!".format(track_id))
    return True


def main(args):
    json_files = glob.glob(os.path.join(args.data_path, '*.json'))

    Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(compute_labels_for_dict)(
            fpath, args.data_path, args.save_path
        ) for fpath in json_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate feature files for multif0 learning.")
    parser.add_argument("data_path",
                        type=str,
                        help="Path where input data lives.")
    parser.add_argument("save_path",
                        type=str,
                        help="Path to save npy files.")
    parser.add_argument("n_jobs",
                        type=int,
                        help="Number of jobs to run in parallel.")
    main(parser.parse_args())
