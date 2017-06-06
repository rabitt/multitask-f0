"""Script to compute training data"""
from __future__ import print_function

import argparse
from joblib import Parallel, delayed
import librosa
import medleydb as mdb
from medleydb import mix
import numpy as np
import os
import scipy
from scipy.signal import upfirdn
from scipy.ndimage import filters
import sox


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


def get_freq_grid():
    """Get the hcqt frequency grid
    """
    (bins_per_octave, n_octaves, _, _, f_min, _) = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies(
        bins_per_octave*n_octaves, f_min, bins_per_octave=bins_per_octave
    )
    return freq_grid


def get_time_grid(n_time_frames):
    """Get the hcqt time grid
    """
    (_, _, _, sr, _, hop_length) = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=sr, hop_length=hop_length
    )
    return time_grid


def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def create_annotation_target(freq_grid, time_grid, annotation_times,
                             annotation_freqs):
    """Create the binary annotation target labels
    """
    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    n_freqs = len(freq_grid)
    n_times = len(time_grid)

    annotation_target = np.zeros((n_freqs, n_times))

    if len(annotation_times) == 0 or len(annotation_freqs) == 0:
        return annotation_target

    annotation_times = np.array(annotation_times)
    annotation_freqs = np.array(annotation_freqs)

    non_zero_idx = np.where(annotation_freqs > 0)[0]
    annotation_times = annotation_times[non_zero_idx]
    annotation_freqs = annotation_freqs[non_zero_idx]

    if len(annotation_times) == 0 or len(annotation_freqs) == 0:
        return annotation_target

    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    idx = annot_time_idx < n_times
    annot_time_idx = annot_time_idx[idx]
    annot_freq_idx = annot_freq_idx[idx]

    idx2 = annot_freq_idx < n_freqs
    annot_time_idx = annot_time_idx[idx2]
    annot_freq_idx = annot_freq_idx[idx2]

    annotation_target[annot_freq_idx, annot_time_idx] = 1

    # create gaussian blur in frequency
    annotation_target_blur = filters.gaussian_filter1d(
        annotation_target, 1, axis=0, mode='constant'
    )
    if len(annot_freq_idx) > 0:
        min_target = np.min(
            annotation_target_blur[annot_freq_idx, annot_time_idx]
        )
    else:
        min_target = 1.0

    annotation_target_blur = annotation_target_blur / min_target
    annotation_target_blur[annotation_target_blur > 1.0] = 1.0

    return annotation_target_blur


def get_annot_activation(times, freqs, mtrack_duration):

    n_time_frames = int(np.ceil(mtrack_duration * float(44100)))
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=44100, hop_length=1
    )
    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])

    annot_time_idx = np.digitize(times, time_bins) - 1
    annot_time_idx = annot_time_idx[annot_time_idx < len(time_grid)]

    freq_complete = np.zeros(time_grid.shape)
    freq_complete[annot_time_idx] = freqs

    annot_activation = np.array(freq_complete > 0, dtype=float)
    annot_activation = upfirdn(np.ones((256, )), annot_activation)

    # blur the edges
    temp = np.zeros(annot_activation.shape)
    temp += annot_activation
    for i in [1, 4, 8, 16, 32, 64, 128]:
        temp[256*i:] += annot_activation[:-(256*i)]
        temp[:-(256*i)] += annot_activation[256*i:]

    annot_activation = np.array(temp > 0, dtype=float)

    annot_activation = np.convolve(
        annot_activation, np.ones((2048,))/2048.0, mode='same'
    )

    return annot_activation


def create_filtered_stem(original_audio, output_path, annot_activation):
    sr = 44100
    y, _ = librosa.load(original_audio, sr=sr, mono=False)

    n_annot = len(annot_activation)
    n_y = len(y[0])

    if n_annot > n_y:
        annot_activation = annot_activation[:n_y]
    elif n_annot < n_y:
        annot_activation = np.append(
            annot_activation, np.zeros((n_y - n_annot, )))

    y_out = np.array(
        [y[0] * annot_activation, y[1] * annot_activation]
    )
    librosa.output.write_wav(output_path, y_out, sr, norm=False)
    return output_path


def activation_filter(mtrack, stem_idx, time_data, freq_data):
    activation = np.array(
        mtrack.activation_conf_from_stem(stem_idx, version='v2')).T
    activation_interpolator = scipy.interpolate.interp1d(
        activation[0], activation[1], fill_value=0.0, bounds_error=False)
    activations = activation_interpolator(time_data)
    freq_data[activations < 0.5] = 0.0
    return freq_data


def get_stem_annotation(stem, mtrack, compute_annot_activity=True,
                        use_estimate=False):
    data = stem.pitch_annotation
    data2 = stem.pitch_estimate_pyin

    if use_estimate:
        data = None

    if data is not None:
        annot = np.array(data).T
        annot_t = annot[0]
        annot_f = annot[1]
        if compute_annot_activity:
            annot_activation = get_annot_activation(
                annot_t, annot_f, mtrack.duration
            )
        else:
            annot_activation = None

        return annot_t, annot_f, annot_activation

    elif data2 is not None:
        annot = np.array(data2).T
        annot_t = annot[0]
        annot_f = annot[1]
        annot_f = activation_filter(mtrack, stem.stem_idx, annot_t, annot_f)
        return annot_t, annot_f, None
    else:
        return None, None, None


def get_all_pitch_annotations(mtrack, compute_annot_activity=False):
    annot_times = []
    annot_freqs = []
    stems_used = []
    stem_annot_activity = {}
    for stem in mtrack.stems.values():

        annot_t, annot_f, annot_activation = get_stem_annotation(
            stem, mtrack
        )

        if annot_t is None:
            continue

        stems_used.append(stem.stem_idx)
        stem_annot_activity[stem.stem_idx] = annot_activation

        annot_times.append(annot_t)
        annot_freqs.append(annot_f)

    if len(annot_times) > 0:
        annot_times = np.concatenate(annot_times)
        annot_freqs = np.concatenate(annot_freqs)

        return annot_times, annot_freqs, stems_used, stem_annot_activity
    else:
        return None, None, None, stem_annot_activity


def get_input_output_pairs(audio_fpath, annot_times, annot_freqs,
                           precomputed_hcqt=None):

    if precomputed_hcqt is None or not os.path.exists(precomputed_hcqt):
        print("    > computing CQT for {}".format(
            os.path.basename(audio_fpath)))
        hcqt = compute_hcqt(audio_fpath)
    else:
        print("    > using precomputed CQT for {}".format(
            os.path.basename(audio_fpath)))
        hcqt = np.load(precomputed_hcqt, mmap_mode='r')

    freq_grid = get_freq_grid()
    time_grid = get_time_grid(len(hcqt[0][0]))

    annot_target = create_annotation_target(
        freq_grid, time_grid, annot_times, annot_freqs
    )

    return hcqt, annot_target, freq_grid, time_grid


def save_data(save_path, prefix, X, Y, f, t):

    input_path = os.path.join(save_path, 'inputs')
    output_path = os.path.join(save_path, 'outputs')
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    np.save(os.path.join(
        input_path, "{}_input.npy".format(prefix)), X.astype(np.float32))
    np.save(os.path.join(
        output_path, "{}_output.npy".format(prefix)), Y.astype(np.float32))

    print("    Saved data for {} to {}".format(prefix, save_path))


def compute_bass(mtrack, save_dir, precomputed_hcqt):
    bass_stems = [
        stem for stem in mtrack.stems.values() if \
        (stem.component == 'bass' \
            or 'electric bass' in stem.instrument \
            or 'double bass' in stem.instrument) \
        and 'm' in stem.f0_type
    ]

    prefix = "{}_bass".format(mtrack.track_id)
    
    input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
    output_path = os.path.join(
        save_dir, 'outputs', "{}_output.npy".format(prefix))
    if os.path.exists(input_path) and os.path.exists(output_path):
        print("    > {} already done!".format(mtrack.track_id))
        return
    
    if len(bass_stems) > 0:
        all_times = []
        all_freqs = []
        print("     > {} Getting times and freqs...".format(mtrack.track_id))
        for bass_stem in bass_stems:
            bass_stem = bass_stems[0]
            
            if bass_stem.pitch_estimate_pyin is None:
                continue

            annot_t, annot_f, _ = get_stem_annotation(
                mtrack, bass_stem.stem_idx, use_estimate=True)

            all_times.append(annot_t)
            all_freqs.append(annot_f)

            
        if len(all_times) == 0:
            return

        times = np.concatenate(all_times)
        freqs = np.concatenate(all_freqs)

        idx = np.where(freqs != 0.0)[0]
        times = times[idx]
        freqs = freqs[idx]

    else:
        print("    {} No bass data".format(mtrack.track_id))
        return
    
    print("     > {} Getting input/output pairs...".format(mtrack.track_id))
    X, Y, f, t = get_input_output_pairs(
        mtrack.mix_path, times, freqs,
        precomputed_hcqt
    )
    save_data(save_dir, prefix, X, Y, f, t)


def compute_vocal(mtrack, save_dir, precomputed_hcqt):
    vocal_stems = [
        stem for stem in mtrack.stems.values() if \
        all([inst in mix.VOCALS for inst in stem.instrument])
    ]
    prefix = "{}_vocal".format(mtrack.track_id)
    
    input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
    output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
    if os.path.exists(input_path) and os.path.exists(output_path):
        print("    > {} already done!".format(mtrack.track_id))
        return
    
    if len(vocal_stems) > 0:
        all_times = []
        all_freqs = []

        for vocal_stem in vocal_stems:
            vocal_stem = vocal_stems[0]

            data = vocal_stem.pitch_annotation
            if data is None:
                data = vocal_stem.pitch_estimate_pyin

            if data is None:
                print("    > skipping stem {} of {} in {}".format(
                    vocal_stem.stem_idx, vocal_stem.instrument, mtrack.track_id))
                continue

            annot_t, annot_f, _ = get_stem_annotation(
                mtrack, vocal_stem.stem_idx, use_estimate=True)

            all_times.append(annot_t)
            all_freqs.append(annot_f)

        if len(all_times) == 0:
            return
            
        times = np.concatenate(all_times)
        freqs = np.concatenate(all_freqs)

        idx = np.where(freqs != 0.0)[0]
        times = times[idx]
        freqs = freqs[idx]

    else:
        print("    {} No vocal data".format(mtrack.track_id))
        return

    X, Y, f, t = get_input_output_pairs(
        mtrack.mix_path, times, freqs,
        precomputed_hcqt
    )
    save_data(save_dir, prefix, X, Y, f, t)


def compute_melody1(mtrack, save_dir, precomputed_hcqt):
    data = mtrack.melody1_annotation
    if data is None:
        print("    {} No melody 1 data".format(mtrack.track_id))
    else:
        prefix = "{}_mel1".format(mtrack.track_id)

        input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
        output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
        if os.path.exists(input_path) and os.path.exists(output_path):
            print("    > {} already done!".format(mtrack.track_id))
            return

        annot = np.array(data).T
        times = annot[0]
        freqs = annot[1]

        idx = np.where(freqs != 0.0)[0]

        times = times[idx]
        freqs = freqs[idx]

        X, Y, f, t = get_input_output_pairs(
            mtrack.mix_path, times, freqs,
            precomputed_hcqt
        )
        save_data(save_dir, prefix, X, Y, f, t)


def compute_melody2(mtrack, save_dir, precomputed_hcqt):
    data = mtrack.melody2_annotation
    if data is None:
        print("    {} No melody 2 data".format(mtrack.track_id))
    else:
        prefix = "{}_mel2".format(mtrack.track_id)

        input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
        output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
        if os.path.exists(input_path) and os.path.exists(output_path):
            print("    > already done!")
            return

        annot = np.array(data).T
        times = annot[0]
        freqs = annot[1]

        idx = np.where(freqs != 0.0)[0]

        times = times[idx]
        freqs = freqs[idx]

        X, Y, f, t = get_input_output_pairs(
            mtrack.mix_path, times, freqs,
            precomputed_hcqt
        )
        save_data(save_dir, prefix, X, Y, f, t)


def compute_melody3(mtrack, save_dir, precomputed_hcqt):
    data = mtrack.melody3_annotation
    if data is None:
        print("   {} No melody 3 data".format(mtrack.track_id))
    else:
        prefix = "{}_mel3".format(mtrack.track_id)

        input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
        output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
        if os.path.exists(input_path) and os.path.exists(output_path):
            print("    > already done!")
            return

        annot = np.array(data).T
        times = annot[0]
        all_freqs = annot[1:]
        time_list = []
        freq_list = []
        for i in range(len(all_freqs)):
            time_list.extend(list(times))
            freq_list.extend(list(all_freqs[i]))

        time_list = np.array(time_list)
        freq_list = np.array(freq_list)
        idx = np.where(freq_list != 0.0)[0]

        time_list = time_list[idx]
        freq_list = freq_list[idx]

        X, Y, f, t = get_input_output_pairs(
            mtrack.mix_path, time_list, freq_list,
            precomputed_hcqt
        )
        save_data(save_dir, prefix, X, Y, f, t)



def compute_multif0_incomplete(mtrack, save_dir,
                               precomputed_hcqt):
    prefix = "{}_multif0_incomplete".format(mtrack.track_id)

    input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
    output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
    if os.path.exists(input_path) and os.path.exists(output_path):
        print("    > already done!")
        return

    times, freqs, _, _ = get_all_pitch_annotations(
        mtrack, compute_annot_activity=False
    )

    if times is not None:

        X, Y, f, t = get_input_output_pairs(
            mtrack.mix_path, times, freqs,
            precomputed_hcqt
        )

        save_data(save_dir, prefix, X, Y, f, t)

    else:
        print("    {} No multif0 data".format(mtrack.track_id))


def compute_multif0_complete(mtrack, save_dir):
    prefix = "{}_multif0_complete".format(mtrack.track_id)

    input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
    output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
    if os.path.exists(input_path) and os.path.exists(output_path):
        print("    > already done!")
        return

    bad_mtrack = False
    for stem in mtrack.stems.values():
        if stem.pitch_estimate_pyin is not None:
            if 'p' in stem.f0_type:
                bad_mtrack = True
    if bad_mtrack:
        print("multitrack has stems with polyphonic instruments")
        return None

    multif0_mix_path = os.path.join(
        save_dir, "{}_multif0_MIX.wav".format(mtrack.track_id)
    )

    if os.path.exists(multif0_mix_path):
        (times, freqs, stems_used,
         stem_annot_activity) = get_all_pitch_annotations(
            mtrack, compute_annot_activity=False
        )
    else:
        (times, freqs, stems_used,
         stem_annot_activity) = get_all_pitch_annotations(
            mtrack, compute_annot_activity=True
        )

    if times is not None:
        for i, stem in mtrack.stems.items():
            unvoiced = all([
                f0_type == 'u' for f0_type in stem.f0_type
            ])
            if unvoiced:
                stems_used.append(i)

        # stems that were manually annotated may not be fully annotated :(
        # silencing out any part of the stem that does not contain
        # annotations just to be safe
        if not os.path.exists(multif0_mix_path):

            alternate_files = {}
            for key in stem_annot_activity.keys():
                new_stem_path = os.path.join(
                    save_dir, "{}_STEM_{}_alt.wav".format(mtrack.track_id, key)
                )
                if not os.path.exists(new_stem_path):
                    create_filtered_stem(
                        mtrack.stems[key].audio_path, new_stem_path,
                        stem_annot_activity[key]
                    )
                alternate_files[key] = new_stem_path


            mix.mix_multitrack(
                mtrack, multif0_mix_path, alternate_files=alternate_files,
                stem_indices=stems_used
            )

        X, Y, f, t = get_input_output_pairs(
            multif0_mix_path, times, freqs, 
        )
        save_data(save_dir, prefix, X, Y, f, t)

    else:
        print("    {} No multif0 data".format(mtrack.track_id))


def compute_features_mtrack(mtrack, save_dir, option,
                            precomputed_hcqt_path, ext='multif0_incomplete'):
    print(mtrack.track_id)
    if precomputed_hcqt_path != '':
        precomputed_hcqt = os.path.join(
            precomputed_hcqt_path, '{}_{}_input.npy'.format(mtrack.track_id, ext)
        )
    else:
        precomputed_hcqt = None

    if option == 'solo_pitch':
        compute_solo_pitch(mtrack, save_dir)
    elif option == 'bass':
        compute_bass(mtrack, save_dir, precomputed_hcqt)
    elif option == 'vocal':
        compute_vocal(mtrack, save_dir, precomputed_hcqt)
    elif option == 'melody1':
        compute_melody1(mtrack, save_dir, precomputed_hcqt)
    elif option == 'melody2':
        compute_melody2(mtrack, save_dir, precomputed_hcqt)
    elif option == 'melody3':
        compute_melody3(mtrack, save_dir, precomputed_hcqt)
    elif option == 'multif0_incomplete':
        compute_multif0_incomplete(
            mtrack, save_dir, precomputed_hcqt
        )
    elif option == 'multif0_complete':
        compute_multif0_complete(mtrack, save_dir)
    else:
        raise ValueError("Invalid value for `option`.")


def main(args):

    if args.use_mdb2:
        dataset_version = ['V1', 'V2', 'EXTRA']
    else:
        dataset_version = ['V1']

    mtracks = mdb.load_all_multitracks(
        dataset_version=dataset_version
    )

    Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(compute_features_mtrack)(
            mtrack, args.save_dir, args.option,
            args.precomputed_hcqt_path
        ) for mtrack in mtracks)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate feature files for multif0 learning.")
    parser.add_argument("save_dir",
                        type=str,
                        help="Path to save npy files.")
    parser.add_argument("option",
                        type=str,
                        help="Type of data to compute. " +
                        "One of 'solo_pitch' 'melody1', 'melody2', 'melody3' " +
                        "'multif0_incomplete', 'multif0_complete'.")
    parser.add_argument("n_jobs",
                        type=int,
                        help="Number of jobs to run in parallel.")
    parser.add_argument("precomputed_hcqt_path",
                        type=str,
                        help="Path to folder with hcqts precomputed")
    parser.add_argument("--use-mdb2",
                        dest='use_mdb2',
                        action='store_true')
    parser.set_defaults(use_mdb2=True)
    main(parser.parse_args())


