"""Script to compute the HCQT on a list of filepaths
"""
from __future__ import print_function

import argparse
import csv
from joblib import Parallel, delayed
import librosa
import numpy as np
import sys


def get_hcqt_params():
    """Static function to store HCQT parameters.
    
    Returns
    -------
    bins_per_octave : int
        Number of bins per octave in HCQT
    n_octaves : int
        Number of octaves in HCQT
    harmonics : list
        List of harmonics to compute in HCQT
    sr : float
        Sample rate to load input audio signal.
    fmin : float
        Minimum frequency in HCQT (in Hz)
    hop_length : int
        Hop length (in samples) at sample rate `sr` of the HCQT

    """
    bins_per_octave = 60
    n_octaves = 6
    harmonics = [1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256
    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length


def compute_hcqt(audio_fpath):
    """Compute the harmonic CQT from a given audio file

    Parameters
    ----------
    audio_fpath : str
        Path to input audio file

    Returns
    -------
    hcqt : np.ndarray
        HCQT as a numpy array
    
    """

    # get parameters and load audio
    (bins_per_octave, n_octaves, harmonics,
     sr, f_min, hop_length) = get_hcqt_params()
    y, fs = librosa.load(audio_fpath, sr=sr)

    # compute cqt for each minimum frequency h*f_min
    cqt_list = []
    shapes = []
    for h in harmonics:
        cqt = librosa.cqt(
            y, sr=fs, hop_length=hop_length, fmin=f_min*float(h),
            n_bins=bins_per_octave*n_octaves,
            bins_per_octave=bins_per_octave
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    # adjust number of time frames across harmonics to the minimum length
    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    # compute log amplitude and normalize between 0 and 1
    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
        np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt


def get_hcqt(input_audio, output_npy):
    """Compute and save an HCQT

    Parameters
    ----------
    input_audio : str
        Path to input audio file
    output_npy : str
        Path to save HCQT npy file

    """
    print("Computing HCQT for {}".format(input_audio))
    try:
        hcqt = compute_hcqt(input_audio)
        np.save(output_npy, hcqt.astype(np.float32))
    except:
        print("Something went wrong for input_audio = {}".format(input_audio))
        print("Unexpected error:", sys.exc_info()[0])


def main(args):
    """Main method to compute HCQTs in parallel
    """

    # expects `args.file_paths` to be a path to a tab delimited file
    # with each line containing an `input_audio` `output_npy` pair.
    file_pairs = []
    with open(args.file_paths, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            file_pairs.append(line)

    # compute HCQTs in parallel
    Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(get_hcqt)(input_audio, output_npy) \
        for input_audio, output_npy in file_pairs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save HCQTs for a set of files.")
    parser.add_argument("file_paths",
                        type=str,
                        help="Tab delimited file containing "
                        "input_audio/output_npy filepaths.")
    parser.add_argument("n_jobs",
                        type=int,
                        help="Number of jobs to run in parallel.")
    main(parser.parse_args())
