"""Scripts to compute multitask features and targets
"""
from __future__ import print_function

import argparse
import glob
import json
import medleydb as mdb
from medleydb import mix
import mir_eval
import numpy as np
import os

import compute_training_data as C
from evaluate import save_singlef0_output


EXCEPTIONS = {
    'AnEndlessSporadic_Anything_STEM_04': 'synthesizer',
    'Anberlin_TheFeelGoodDrag_STEM_04': 'synthesizer',
    'ArcadeFire_BlackMirror_STEM_02': 'acoustic guitar',
    'ArcadeFire_BlackMirror_STEM_07': 'acoustic guitar',
    'BillyIdol_WhiteWedding_STEM_05': 'distorted electric guitar',
    'Blink182_AllTheSmallThings_STEM_05': 'clean electric guitar',
    'Blondie_OneWayOrAnother_STEM_06': 'distorted electric guitar',
    'BlueOysterCult_DontFearTheReaper_STEM_05': 'clean electric guitar',
    'LilyAllen_NotFair_STEM_04': 'clean electric guitar',
    'TheAllAmericanRejects_DirtyLittleSecret_STEM_04': 'distorted electric guitar',
    'TheAllmanBrothersBand_RamblinMan_STEM_06': 'clean electric guitar',
    'TheLastGoodnight_PicturesOfYou_STEM_02': 'distorted_electric guitar'
}


def multif0_to_timefreq(times, freqs):
    """Unroll a multif0 annotation of the form (t, [f1, f2, f3])
    to a list of (t, f) where t may be repeated.

    Parameters
    ----------
    times : list
        Time stamps
    freqs : list of lists
        all frequencies for a given time stamp

    Returns
    -------
    t_unrolled : list
        Unrolled time stamps
    f_unrolled : list
        Unrolled frequency values

    """
    t_unrolled = []
    f_unrolled = []
    for t, f_list in zip(times, freqs):
        for f in f_list:
            if f == 0:
                continue
            t_unrolled.append(t)
            f_unrolled.append(f)
    return t_unrolled, f_unrolled


def get_replace_info(mtrack, replace_path):
    """Go through repalced stems and get annotation and mixing info.

    Parameters
    ----------
    mtrack : MultiTrack
        Medleydb multitrack object
    replace_path : str
        Path to where the resynthesized audio and annotations live

    Returns
    -------
    replace_stem_annotations : dictionary
        Dictionary keyed by stem id mapping to 'times', 'freqs', and 'tags'
        for any relevant annotations.
    replace_altindices : dictionary
        Dictionary keyed by stem id mapping to the replaced path.
        Used by mix.mix_multitrack for replacing stems.
    stem_indices : list
        List of stem indices for replaced stems

    """
    # glob for all replaced stems for this multitrack
    replace_stems = glob.glob(os.path.join(
        replace_path, '{}*_replace.wav'.format(mtrack.track_id)
    ))

    # initialize
    stem_indices = []
    replace_stem_annotations = {}
    replace_altindices = {}
    
    # loop over resynthesized stems
    for stem_path in replace_stems:
        # path where annotation should live
        annot_path = os.path.join(
            replace_path,
            "{}_vamp_pyin_pyin_smoothedpitchtrack.csv".format(
                os.path.basename(stem_path).split('.')[0])
        )

        # if annotation doesn't exist, cry!
        if not os.path.exists(annot_path):
            print("[Warning] Couldn't find annotation for {}".format(stem_path))
            continue

        # parse stem index from file name
        stem_id = int(os.path.basename(stem_path).split('_')[3].split('.')[0])

        # load resynth annotation
        times, freqs = mir_eval.io.load_ragged_time_series(
            annot_path, delimiter=','
        )
        annot_t, annot_f = multif0_to_timefreq(times, freqs)
        tags = ['multif0', 'vocal', 'melody']
        stem_indices.append(stem_id)

        # add annotation to dictionary
        replace_stem_annotations[stem_id] = {
            'times': annot_t, 'freqs': annot_f, 'tags': tags
        }

        # add resynth file to replacement dictionary
        replace_altindices[stem_id] = stem_path

    return replace_stem_annotations, replace_altindices, stem_indices


def get_resynth_info(mtrack, resynth_path, stem_indices):
    """Go through resynthesized stems and get annotation and mixing info.

    Parameters
    ----------
    mtrack : MultiTrack
        Medleydb multitrack object
    resynth_path : str
        Path to where the resynthesized audio and annotations live
    stem_indices : list
        List of indices already used

    Returns
    -------
    resynth_stem_annotations : dictionary
        Dictionary keyed by stem id mapping to 'times', 'freqs', and 'tags'
        for any relevant annotations.
    resynth_altindices : dictionary
        Dictionary keyed by stem id mapping to the resynthesized path.
        Used by mix.mix_multitrack for replacing stems.
    stem_indices_guitar : list
        List of stem indices containing any kind of resynthesized guitar
    stem_indices_piano : list
        List of stem indices containing any kind of resynthesized piano

    """
    # glob for all resynth stems for this multitrack
    resynth_stems = glob.glob(os.path.join(
        resynth_path, '{}*_resynth.wav'.format(mtrack.track_id)
    ))

    # initialize
    stem_indices_piano = []
    stem_indices_guitar = []
    resynth_stem_annotations = {}
    resynth_altindices = {}
    
    # loop over resynthesized stems
    for stem_path in resynth_stems:
        # path where annotation should live
        annot_path = os.path.join(
            resynth_path,
            "{}.txt".format(os.path.basename(stem_path).split('.')[0])
        )

        # parse stem index from file name
        stem_id = int(os.path.basename(stem_path).split('_')[3].split('.')[0])
        stem = mtrack.stems[stem_id]

        if stem_id in stem_indices:
            continue

        # if annotation doesn't exist, cry!
        if not os.path.exists(annot_path):
            print("[Warning] Couldn't find annotation for {}".format(stem_path))
            continue

        # load resynth annotation
        times, freqs = mir_eval.io.load_ragged_time_series(annot_path)
        annot_t, annot_f = multif0_to_timefreq(times, freqs)
        tags = ['multif0']

        # apply tags based on whether instrument is piano or guitar
        basename = os.path.basename(stem.audio_path.split('.')[0])
        if basename in EXCEPTIONS.keys():
            instrument = EXCEPTIONS[basename]
        else:
            instrument = stem.instrument

        if 'piano' in instrument:
            tags.append('piano')
            stem_indices_piano.append(stem_id)
        elif ('electric piano' in instrument or
              'synthesizer' in instrument):
            stem_indices_piano.append(stem_id)
        elif ('acoustic guitar' in instrument or
              'clean electric guitar' in instrument or
              'distorted electric guitar' in instrument):
            tags.append('guitar')
            stem_indices_guitar.append(stem_id)
        else:
            print("[Warning] resynth stem is instrument {}! skipping".format(
                instrument))
            continue

        # add annotation to dictionary
        resynth_stem_annotations[stem_id] = {
            'times': annot_t, 'freqs': annot_f, 'tags': tags
        }

        # add resynth file to replacement dictionary
        resynth_altindices[stem_id] = stem_path

    return (resynth_stem_annotations, resynth_altindices,
            stem_indices_guitar, stem_indices_piano)


def get_orig_stem_info(mtrack, stem_indices):
    """Go through original stems and get annotation and mixing info.

    Parameters
    ----------
    mtrack : MultiTrack
        Medleydb multitrack object
    stem_indices : list
        List of stem indices already included

    Returns
    -------
    orig_stem_annotations : dictionary
        Dictionary keyed by stem id mapping to 'times', 'freqs', and 'tags'
        for any relevant annotations.
    stem_annot_activity : dictionary
        Dictionary keyed by stem id mapping to annotation activation information
    orig_stem_indices : list
        List of stem indices to include in mix.

    """
    orig_stem_indices = []
    stem_annot_activity = {}
    orig_stem_annotations = {}

    # go through the rest of the stems
    for stem in mtrack.stems.values():

        # skip this stem if it was resynthesized
        if stem.stem_idx in stem_indices:    
            continue

        # skip this stem if it has more than one instrument
        if len(stem.instrument) > 1 or len(stem.f0_type) > 1:
            continue

        # skip this stem if it is a polyphonic instrument
        if stem.f0_type[0] == 'p':
            continue

        # if stem is unpitched, just add it to the mix
        if stem.f0_type[0] == 'u':
            orig_stem_indices.append(stem.stem_idx)

        # if stem is mono, add it to the mix and get its annotation
        if len(stem.instrument) == 1 and stem.f0_type[0] == 'm':
            orig_stem_indices.append(stem.stem_idx)
            annot_t, annot_f, annot_activation = C.get_stem_annotation(
                stem, mtrack
            )
            stem_annot_activity[stem.stem_idx] = annot_activation
            tags = ['multif0']

            if stem.instrument[0] in mix.VOCALS:
                tags.append('vocal')

            if stem.component == 'melody':
                tags.append('melody')

            if stem.component == 'bass':
                tags.append('bass')

            orig_stem_annotations[stem.stem_idx] = {
                'times': annot_t, 'freqs': annot_f, 'tags': tags
            }

    return orig_stem_annotations, stem_annot_activity, orig_stem_indices


def save_annotation(times, freqs, save_path):
    """ Save an annotation to a filepath or return None if annotation is
    empty.

    Parameters
    ----------
    times : list
        List of times
    freqs : list
        List of freqs
    save_path : str
        Path to save file

    Returns
    -------
    output : str or None
        If times/freqs are not empty, returns save_path. Otherwise
        returns None.
    """
    if len(times) > 0:
        # using singlef0 save here because we have unwrapped multif0s
        save_singlef0_output(times, freqs, save_path)
        return save_path
    else:
        return None


def create_annotations(save_dir, track_id, stem_annotations):
    """Create a dictionary of annotations by type, given a list of annotations
    by stem.

    Parameters
    ----------
    save_dir : str
        Path to eventually save each annotation.
    track_id : str
        Medleydb trackid
    stem_annotations : dictionary
        Dictionary keyed by stem id with values 'times', 'freqs' and 'tags'

    Returns
    -------
    annotations : dictionary
        Dictionary keyed by annotation type (e.g. 'vocal') with values
        'times' 'freqs' and 'path'.

    """

    # create initial annotations dictionary
    annotations = {
        'multif0': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_multif0_annotation.txt".format(track_id))
        },
        'multif0_noguitar': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_multif0_noguitar_annotation.txt".format(track_id))
        },
        'multif0_nosynth': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_multif0_nosynth_annotation.txt".format(track_id))
        },
        'melody': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_melody_annotation.txt".format(track_id))
        },
        'vocal': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_vocal_annotation.txt".format(track_id))
        },
        'bass': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_bass_annotation.txt".format(track_id))
        },
        'piano': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_piano_annotation.txt".format(track_id))
        },
        'guitar': {
            'times': [], 'freqs': [], 'path': os.path.join(
                save_dir, "{}_guitar_annotation.txt".format(track_id))
        }
    }

    # loop over each stem annotation and add it to corresponding annotation
    # types (e.g. stems with melody tag are added to the melody annotation)
    for key in sorted(stem_annotations.keys()):
        annot_dict = stem_annotations[key]
        tags = annot_dict['tags']

        # all stems should have the 'multif0' tag
        if 'multif0' in tags:
            annotations['multif0']['times'].extend(annot_dict['times'])
            annotations['multif0']['freqs'].extend(annot_dict['freqs'])

        # if stem is guitar, add it to guitar
        if 'guitar' in tags:
            annotations['guitar']['times'].extend(annot_dict['times'])
            annotations['guitar']['freqs'].extend(annot_dict['freqs'])
        # if stem is not guitar add it to the multif0 no guitar annotation
        else:
            annotations['multif0_noguitar']['times'].extend(annot_dict['times'])
            annotations['multif0_noguitar']['freqs'].extend(annot_dict['freqs'])

        # if stem is piano add to piano annotation
        if 'piano' in tags:
            annotations['piano']['times'].extend(annot_dict['times'])
            annotations['piano']['freqs'].extend(annot_dict['freqs'])

        # if stem is not synthesized (i.e. not piano or guitar) add it to
        # the nosynth annotation
        if 'piano' not in tags and 'guitar' not in tags:
            annotations['multif0_nosynth']['times'].extend(annot_dict['times'])
            annotations['multif0_nosynth']['freqs'].extend(annot_dict['freqs'])

        # add melody stems to melody annotation
        if 'melody' in tags:
            annotations['melody']['times'].extend(annot_dict['times'])
            annotations['melody']['freqs'].extend(annot_dict['freqs'])

        # add vocal stems to vocal annotation
        if 'vocal' in tags:
            annotations['vocal']['times'].extend(annot_dict['times'])
            annotations['vocal']['freqs'].extend(annot_dict['freqs'])

        # add bass stems to bass annotation
        if 'bass' in tags:
            annotations['bass']['times'].extend(annot_dict['times'])
            annotations['bass']['freqs'].extend(annot_dict['freqs'])

    return annotations


def create_annotation_save_pairs(annotations, mix_path, mix_path_noguitar,
                                 mix_path_nosynth):
    """Create a dictionary that maps an audio file to its corresponding
    multitask annotations.

    Parameters
    ----------
    annotations : dictionary
        Dictionary mapping annotation type to 'times', 'freqs', and 'path'
        to save.
    mix_path : str
        Path to full multif0 mix
    mix_path_noguitar : str
        Path to no guitar multif0 mix
    mix_path_nosynth : str
        Path to no synthesized stems multif0 mix

    Returns
    -------
    audio_annot_pairs : dictionary
        Dictionary mapping audio path to annotation paths by type.

    """
    audio_annot_pairs = {
        mix_path: {},
        mix_path_noguitar: {},
        mix_path_nosynth: {}
    }

    for annot_type in annotations.keys():
        output = save_annotation(
            annotations[annot_type]['times'],
            annotations[annot_type]['freqs'],
            annotations[annot_type]['path']
        )
        if annot_type == 'multif0':
            audio_annot_pairs[mix_path]['multif0'] = output
        elif annot_type == 'multif0_noguitar':
            audio_annot_pairs[mix_path_noguitar]['multif0'] = output
        elif annot_type == 'multif0_nosynth':
            audio_annot_pairs[mix_path_nosynth]['multif0'] = output
        elif annot_type == 'guitar':
            audio_annot_pairs[mix_path]['guitar'] = output
            audio_annot_pairs[mix_path_noguitar]['guitar'] = None
            audio_annot_pairs[mix_path_nosynth]['guitar'] = None
        elif annot_type == 'piano':
            audio_annot_pairs[mix_path]['piano'] = output
            audio_annot_pairs[mix_path_noguitar]['piano'] = output
            audio_annot_pairs[mix_path_nosynth]['piano'] = None
        else:
            audio_annot_pairs[mix_path][annot_type] = output
            audio_annot_pairs[mix_path_noguitar][annot_type] = output
            audio_annot_pairs[mix_path_nosynth][annot_type] = output

    return audio_annot_pairs


def generate_filtered_stems(stem_annot_activity, mtrack, save_dir):
    """Create filtered stems for stems with annotation activity info.

    Parameters
    ----------
    stem_annot_activity : dictionary
        Dictionary mapping stem_id to annotation activity information
    mtrack : MultiTrack
        medleydb MultiTrack object
    save_dir : str
        Path to save new stems.

    Returns
    -------
    filtered_altfiles : dictionary
        Dictionary mapping stem_id to a path where the new stem is saved.

    """
    filtered_altfiles = {}
    
    # for each stem with an annotation filter, create filtered stem
    for key in stem_annot_activity.keys():
        
        if stem_annot_activity[key] is None:
            continue

        new_stem_path = os.path.join(
            save_dir, "{}_STEM_{}_alt.wav".format(mtrack.track_id, key)
        )

        if not os.path.exists(new_stem_path):
            C.create_filtered_stem(
                mtrack.stems[key].audio_path, new_stem_path,
                stem_annot_activity[key]
            )

        filtered_altfiles[key] = new_stem_path

    return filtered_altfiles


def create_mixes(mtrack, mix_path, mix_path_noguitar, mix_path_nosynth,
                 stem_indices, stem_indices_guitar, stem_indices_piano,
                 altfiles):
    """Render artificial mixes to `mix_path', `mix_path_noguitar', and
    `mix_path_nosynth'.

    Parameters
    ----------
    mtrack : MultiTrack
        medleydb MultiTrack object
    mix_path : str
        Path to save full multif0 mix
    mix_path_noguitar : str
        Path to save no guitar multif0 mix
    mix_path_nosynth : str
        Path to save no synthesized stems multif0 mix
    stem_indices : list
        List of stems to include in the full mix
    stem_indices_guitar : list
        List of guitar stems
    stem_indices_piano : list
        List of piano stems
    altfiles : dict
        Dictionary of replacement files mapping stem id to new path.

    Returns
    -------
    mix_filepaths : list
        List of filepaths included in the full mix
    mix_noguitar_filepaths : list
        List of filepaths included in the no guitar mix
    mix_nosynth_filepaths : list
        List of filepaths included in the no resynth mix

    """
    # create resynth mix
    mix_filepaths, _ = mix.mix_multitrack(
        mtrack, mix_path, stem_indices=stem_indices, alternate_files=altfiles
    )
    # create no guitar and no synth mixes
    stem_indices_noguitar = [
        s for s in stem_indices if s not in stem_indices_guitar
    ]
    stem_indices_nosynth = [
        s for s in stem_indices_noguitar if s not in stem_indices_piano
    ]
    altfiles_noguitar = {
        k: v for k, v in altfiles.items() if k in stem_indices_noguitar
    }
    altfiles_nosynth = {
        k: v for k, v in altfiles.items() if k in stem_indices_nosynth
    }
    mix_noguitar_filepaths, _ = mix.mix_multitrack(
        mtrack, mix_path_noguitar, stem_indices=stem_indices_noguitar,
        alternate_files=altfiles_noguitar
    )
    mix_nosynth_filepaths, _ = mix.mix_multitrack(
        mtrack, mix_path_nosynth, stem_indices=stem_indices_nosynth,
        alternate_files=altfiles_nosynth
    )
    return mix_filepaths, mix_noguitar_filepaths, mix_nosynth_filepaths


def create_complete_resynth_mix(mtrack, resynth_path, replace_path, save_dir):
    """Create resynthesized mixes and all corresponding annotations

    Audio files:
    - (A) Multif0 remix with synth guitar + synth piano
    - (B) Multif0 remix with synth piano
    - (C) Multif0 remix
    - (D) Original track

    Annotations:
    filename : description (corresponding audio file)
    - Artist_Track_multif0_annotation.txt : multif0 + synth piano/guitar (A)
    - Artist_Track_multif0_noguiar_annotation.txt : multif0 + synth piano (B)
    - Artist_Track_multif0_nosynth_annotation.txt : multif0 (C)
    - Artist_Track_melody_annotation.txt : all melody f0s (A,B,C,[D])
    - Artist_Track_vocal_annotation.txt : all vocal f0s (A,B,C,[D])
    - Artist_Track_bass_annotation.txt : all bass f0s (A,B,C,[D])
    - Artist_Track_piano_annotation.txt : all piano f0s (A,B)
    - Artist_Track_guitar_annotation.txt : all guitar f0s (A)

    Parameters
    ----------
    mtrack : MultiTrack
        medleydb MultiTrack object
    resynth_path : str
        Path where resynthesized files live
    replace_path : str
        Path where replacement files live
    save_dir : str
        Path to save output.

    Returns
    -------
    audio_annot_pairs : dictionary
        Dictionary mapping audio files to annotation files by type.

    """

    # do nothing if track has bleed
    if mtrack.has_bleed:
        return None

    # mix audio save paths
    mix_path = os.path.join(
        save_dir, "{}_MIX_complete_resynth.wav".format(mtrack.track_id)
    )
    mix_path_noguitar = os.path.join(
        save_dir, "{}_MIX_complete_noguitar.wav".format(mtrack.track_id)
    )
    mix_path_nosynth = os.path.join(
        save_dir, "{}_MIX_complete_nosynth.wav".format(mtrack.track_id)
    )

    # define common structures
    stem_indices = []
    altfiles = {}
    stem_annotations = {}

    # get all annotation and index info from resynthesized stems
    (replace_stem_annotations, replace_altindices,
     stem_indices_replace) = get_replace_info(
         mtrack, replace_path
     )

    stem_indices.extend(stem_indices_replace)
    for key, value in replace_stem_annotations.items():
        stem_annotations[key] = value
    for key, value in replace_altindices.items():
        altfiles[key] = value

    # get all annotation and index info from resynthesized stems
    (resynth_stem_annotations, resynth_altindices,
     stem_indices_guitar, stem_indices_piano) = get_resynth_info(
         mtrack, resynth_path, stem_indices
     )

    stem_indices.extend(stem_indices_piano)
    stem_indices.extend(stem_indices_guitar)
    for key, value in resynth_stem_annotations.items():
        stem_annotations[key] = value
    for key, value in resynth_altindices.items():
        altfiles[key] = value

    # get all annotation and index info from remaining original stems
    (orig_stem_annotations, stem_annot_activity,
     orig_stem_indices) = get_orig_stem_info(mtrack, stem_indices)

    # fill info to common structures
    stem_indices.extend(orig_stem_indices)
    for key, value in orig_stem_annotations.items():
        stem_annotations[key] = value

    # create annotation dictionary
    annotations = create_annotations(
        save_dir, mtrack.track_id, stem_annotations
    )

    # save annotation and create pairs
    audio_annot_pairs = create_annotation_save_pairs(
        annotations, mix_path, mix_path_noguitar, mix_path_nosynth
    )

    # create new versions of stems with annotation filters
    filtered_altfiles = generate_filtered_stems(
        stem_annot_activity, mtrack, save_dir
    )
    for key, value in filtered_altfiles.items():
        altfiles[key] = value

    # make sure there is a least one stem left in the mix
    if len(stem_indices) == 0:
        print("{} had no stems after filtering :( ".format(mtrack.track_id))
        return None

    # generate mixes
    create_mixes(
        mtrack, mix_path, mix_path_noguitar, mix_path_nosynth,
        stem_indices, stem_indices_guitar, stem_indices_piano, altfiles
    )

    return audio_annot_pairs


def get_annotation_mono(mtrack, stem_list):
    """Get annotation for a subset of stems if all stems are mono

    Parameters
    ----------
    mtrack : MultiTrack
        medleydb MultiTrack object
    stem_list : list
        list of Track objects

    Returns
    -------
    times : list or None
        list of times or None
    freqs : list or None
        list of freqs or None

    """
    # if no stems, the annotation is empty
    if len(stem_list) == 0:
        times = []
        freqs = []

    # otherwise, check if all stems are mono
    else:
        all_mono = True
        for stem in stem_list:
            if len(stem.instrument) > 1:
                all_mono = False
            elif stem.f0_type[0] != 'm':
                all_mono = False

        # if all stems are mono add the annotation to the mix
        if all_mono:
            times = []
            freqs = []
            for stem in stem_list:
                annot_t, annot_f, _ = C.get_stem_annotation(
                    stem, mtrack, use_estimate=True
                )

                # if there is no annotation return None
                if annot_t is None or annot_f is None:
                    return None, None

                times.extend(annot_t)
                freqs.extend(annot_f)
        else:
            times = None
            freqs = None

    return times, freqs


def get_fullmix_annotations(mtrack, save_dir):
    """Get annotations corresponding to original medleydb mixes.

    Parameters
    ----------
    mtrack : MultiTrack
        A medleydb MultiTrack object.
    save_dir : str
        Path to save annotation files.

    Returns
    -------
    audio_annot_pairs : dictionary
        Dictionary mapping audio files to annotation files by type.

    """
    audio_annot_pairs = {mtrack.mix_path: {}}

    melody_stems = []
    vocal_stems = []
    bass_stems = []
    guitar_stems = []
    piano_stems = []

    guitars = ['acoustic guitar', 'clean electric guitar', 'distorted guitar']

    for stem in mtrack.stems.values():
        if any(inst in mix.VOCALS for inst in stem.instrument):
            vocal_stems.append(stem)
        if 'Unlabeled' in stem.instrument and stem.component == 'melody':
            vocal_stems.append(stem)
        if stem.component == 'bass':
            bass_stems.append(stem)
        if stem.component == 'melody':
            melody_stems.append(stem)
        if any(inst in guitars for inst in stem.instrument):
            guitar_stems.append(stem)
        if any(inst == 'piano' for inst in stem.instrument):
            piano_stems.append(stem)

    # use melody if there is melody or none
    if mtrack.dataset_version == 'V1':
        if mtrack.melody3_annotation is not None:
            annot = np.array(mtrack.melody3_annotation).T
            melody_times, melody_freqs = multif0_to_timefreq(
                annot[0], annot[1:])
        else:
            melody_times = []
            melody_freqs = []
    else:
        melody_times, melody_freqs = get_annotation_mono(mtrack, melody_stems)

    if melody_times is not None:
        output = save_annotation(
            melody_times,
            melody_freqs,
            os.path.join(save_dir, '{}_MIX_melody.txt'.format(mtrack.track_id))
        )
        audio_annot_pairs[mtrack.mix_path]['melody'] = output

    # use vocals if all vocals are mono or there are none
    vocal_times, vocal_freqs = get_annotation_mono(mtrack, vocal_stems)
    if vocal_times is not None:
        output = save_annotation(
            vocal_times,
            vocal_freqs,
            os.path.join(save_dir, '{}_MIX_vocal.txt'.format(mtrack.track_id))
        )
        audio_annot_pairs[mtrack.mix_path]['vocal'] = output

    # use bass if all bass is mono or there are none
    bass_times, bass_freqs = get_annotation_mono(mtrack, bass_stems)
    if bass_times is not None:
        output = save_annotation(
            bass_times,
            bass_freqs,
            os.path.join(save_dir, '{}_MIX_bass.txt'.format(mtrack.track_id))
        )
        audio_annot_pairs[mtrack.mix_path]['bass'] = output

    # mark that there's no piano/guitar if there are no stems with
    # those instruments
    if len(piano_stems) == 0:
        audio_annot_pairs[mtrack.mix_path]['piano'] = None

    if len(guitar_stems) == 0:
        audio_annot_pairs[mtrack.mix_path]['guitar'] = None

    return audio_annot_pairs


def get_all_audio_annot_pairs(mtrack, save_dir, resynth_path, replace_path):
    """For a given multitrack get all types of mixes and corresponding
    annotations, and save a json file with all info.

    Parameters
    ----------
    mtrack : MultiTrack
        medleydb MultiTrack object.
    save_dir : str
        Path to save json output file.
    resynth_path : str
        Path to where resynthesized stems live.
    replace_path : str
        Path to where replaced stems live

    Returns
    -------
    json_path : str
        Path to saved json file

    """
    print("    Resynth annotations and mixing...")
    resynth_pairs = create_complete_resynth_mix(
        mtrack, resynth_path, replace_path, save_dir)
    print("    Fullmix annotations")
    fullmix_pairs = get_fullmix_annotations(mtrack, save_dir)
    all_pairs = {}
    for key, value in resynth_pairs.items():
        all_pairs[key] = value

    for key, value in fullmix_pairs.items():
        all_pairs[key] = value

    json_path = os.path.join(
        save_dir, "{}_training_pairs.json".format(mtrack.track_id)
    )

    with open(json_path, 'w') as fhandle:
        json.dump(all_pairs, fhandle, indent=2)

    return json_path


def main(args):
    mtracks = mdb.load_all_multitracks(
        dataset_version=['V1', 'V2', 'EXTRA'])
    for mtrack in mtracks:
        if mtrack.has_bleed:
            continue

        print("Processing {}...".format(mtrack.track_id))

        if os.path.exists(os.path.join(args.save_dir,
                          "{}_training_pairs.json".format(mtrack.track_id))):
            print("    already done!")
            continue
        
        json_path = get_all_audio_annot_pairs(
            mtrack, args.save_dir, args.resynth_path, args.replace_path
        )
        print("...saved to {}".format(json_path))
        print("")

    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate feature files for multif0 learning.")
    parser.add_argument("save_dir",
                        type=str,
                        help="Path to save npy files.")
    parser.add_argument("resynth_path",
                        type=str,
                        help="resynth path")
    parser.add_argument("replace_path",
                        type=str,
                        help="replace path")
    main(parser.parse_args())
