import unittest
import os
import numpy as np

import medleydb as mdb
from deepsalience import compute_multitask_data as CMD

class TestMultiF0ToTimeFreq(unittest.TestCase):

    def test_default(self):
        times = [0.0, 0.1, 0.2, 0.8]
        freqs = [[], [200.1, 207.7], [], [82.1]]
        actual_times, actual_freqs = CMD.multif0_to_timefreq(times, freqs)
        expected_times = [0.1, 0.1, 0.8]
        expected_freqs = [200.1, 207.7, 82.1]
        self.assertEqual(expected_times, actual_times)
        self.assertEqual(expected_freqs, actual_freqs)


class TestGetReplaceInfo(unittest.TestCase):

    def test_one(self):
        replace_path = os.path.join(
            "/Users/rabitt/Dropbox/piano_guitar_resynth",
            "vocalists_replace"
        )
        mtrack = mdb.MultiTrack('AHa_TakeOnMe')
        (actual_annotations,
         actual_altindices,
         actual_indicies) = CMD.get_replace_info(mtrack, replace_path)

        expected_annotations = {  
            2: {
                'tags': ['multif0', 'vocal', 'melody'],
                'times': [11.104943311],
                'freqs': [150.758]
            }
        }
        expected_altindices = {
            2: os.path.join(replace_path, 'AHa_TakeOnMe_STEM_02_replace.wav')
        }
        expected_indices = [2]

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_altindices, actual_altindices)
        self.assertEqual(expected_indices, actual_indicies)

    def test_two(self):
        mtrack = mdb.MultiTrack('ArcadeFire_BlackMirror')
        replace_path = os.path.join(
            "/Users/rabitt/Dropbox/piano_guitar_resynth",
            "vocalists_replace"
        )
        (actual_annotations,
         actual_altindices,
         actual_indicies) = CMD.get_replace_info(mtrack, replace_path)

        expected_annotations = {}
        expected_altindices = {}
        expected_indices = []

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_altindices, actual_altindices)
        self.assertEqual(expected_indices, actual_indicies)


class TestGetResynthInfo(unittest.TestCase):

    def test_one(self):
        resynth_path = '/Users/rabitt/Dropbox/piano_guitar_resynth/resynth'
        mtrack = mdb.MultiTrack('AHa_TakeOnMe')
        (actual_annotations,
         actual_altindices,
         actual_indicies_guitar,
         actual_indices_piano) = CMD.get_resynth_info(
             mtrack, resynth_path, [2])

        expected_annotations = {  
            1: {
                'tags': ['multif0'],
                'times': [119.4550566893424],
                'freqs': [830.60939515989025]
            },
            3: {
                'tags': ['multif0'],
                'times': [11.435827664399094],
                'freqs': [123.47082531403103]
            },
            4: {
                'tags': ['multif0', 'guitar'],
                'times': [49.458503401360545],
                'freqs': [293.66476791740757]
            },
            6: {
                'tags': ['multif0'],
                'times': [5.5495691609977325],
                'freqs': [246.94165062806206]
            }
        }
        expected_altindices = {
            1: os.path.join(resynth_path, 'AHa_TakeOnMe_STEM_01_resynth.wav'),
            3: os.path.join(resynth_path, 'AHa_TakeOnMe_STEM_03_resynth.wav'),
            4: os.path.join(resynth_path, 'AHa_TakeOnMe_STEM_04_resynth.wav'),
            6: os.path.join(resynth_path, 'AHa_TakeOnMe_STEM_06_resynth.wav')
        }
        expected_indices_guitar = [4]
        expected_indices_piano = [1, 3, 6]

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_altindices, actual_altindices)
        self.assertEqual(expected_indices_guitar, actual_indicies_guitar)
        self.assertEqual(expected_indices_piano, actual_indices_piano)

    def test_two(self):
        resynth_path = '/Users/rabitt/Dropbox/piano_guitar_resynth/resynth'
        mtrack = mdb.MultiTrack('Adele_SomeoneLikeYou')
        (actual_annotations,
         actual_altindices,
         actual_indicies_guitar,
         actual_indices_piano) = CMD.get_resynth_info(
             mtrack, resynth_path, [2])

        expected_annotations = {  
            1: {
                'tags': ['multif0', 'piano'],
                'times': [0.83591836734693881],
                'freqs': [164.81377845643496]
            }
        }
        expected_altindices = {
            1: os.path.join(
                resynth_path, 'Adele_SomeoneLikeYou_STEM_01_resynth.wav')
        }
        expected_indices_guitar = []
        expected_indices_piano = [1]

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_altindices, actual_altindices)
        self.assertEqual(expected_indices_guitar, actual_indicies_guitar)
        self.assertEqual(expected_indices_piano, actual_indices_piano)

    def test_three(self):
        mtrack = mdb.MultiTrack('ArcadeFire_BlackMirror')
        resynth_path = '/Users/rabitt/Dropbox/piano_guitar_resynth/resynth'
        stem_indices = []
        (actual_annotations,
         actual_altindices,
         actual_indicies_guitar,
         actual_indices_piano) = CMD.get_resynth_info(
             mtrack, resynth_path, stem_indices)

        expected_annotations = {  
            2: {
                'tags': ['multif0', 'guitar'],
                'times': [10.832108843537416],
                'freqs': [116.54094037952248]
            },
            7: {
                'tags': ['multif0', 'guitar'],
                'times': [11.122358276643991],
                'freqs': [116.54094037952248]
            }
        }
        expected_altindices = {
            2: os.path.join(
                resynth_path, 'ArcadeFire_BlackMirror_STEM_02_resynth.wav'),
            7: os.path.join(
                resynth_path, 'ArcadeFire_BlackMirror_STEM_07_resynth.wav')
        }
        expected_indices_guitar = [2, 7]
        expected_indices_piano = []

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_altindices, actual_altindices)
        self.assertEqual(expected_indices_guitar, actual_indicies_guitar)
        self.assertEqual(expected_indices_piano, actual_indices_piano)


class TestGetOrigStemInfo(unittest.TestCase):

    def test_one(self):
        mtrack = mdb.MultiTrack('AcDc_BackInBlack')
        stem_indices = [2, 4]
        (actual_annotations,
         actual_annot_activity,
         actual_indices) = CMD.get_orig_stem_info(mtrack, stem_indices)

        expected_annotations = {  
            3: {
                'tags': ['multif0', 'bass'],
                'times': [5.294149660],
                'freqs': [85.2297]
            },
            6: {
                'tags': ['multif0', 'vocal', 'melody'],
                'times': [25.094965986],
                'freqs': [689.325]
            }
        }
        expected_annot_activity = {3: None, 6: None}
        expected_indices = [3, 5, 6]

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_annot_activity, actual_annot_activity)
        self.assertEqual(expected_indices, actual_indices)

    def test_two(self):
        mtrack = mdb.MultiTrack('ArcadeFire_BlackMirror')
        stem_indices = [2, 7]
        (actual_annotations,
         actual_annot_activity,
         actual_indices) = CMD.get_orig_stem_info(mtrack, stem_indices)

        expected_annotations = {  
            4: {
                'tags': ['multif0', 'bass'],
                'times': [0.058049887],
                'freqs': [0.0]
            },
            6: {
                'tags': ['multif0', 'vocal', 'melody'],
                'times': [26.790022676],
                'freqs': [162.773]
            }
        }
        expected_annot_activity = {4: None, 6: None}
        expected_indices = [3, 4, 6]

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_annot_activity, actual_annot_activity)
        self.assertEqual(expected_indices, actual_indices)

    def test_three(self):
        mtrack = mdb.MultiTrack('MusicDelta_BebopJazz')
        stem_indices = [3]
        (actual_annotations,
         actual_annot_activity,
         actual_indices) = CMD.get_orig_stem_info(mtrack, stem_indices)

        expected_annotations = {
            2: {
                'tags': ['multif0', 'bass'],
                'times': [0.632743764],
                'freqs': [86.5701]
            },
            4: {
                'tags': ['multif0', 'melody'],
                'times': [0.429569160],
                'freqs': [190.852]
            },
            5: {
                'tags': ['multif0', 'melody'],
                'times': [0.400544217],
                'freqs': [380.532]
            }
        }
        expected_annot_activity = {2: None, 4: [0.5], 5: [0.5]}
        expected_indices = [1, 2, 4, 5]

        self.assertEqual(expected_annotations.keys(), actual_annotations.keys())
        for k in actual_annotations.keys():
            self.assertEqual(
                expected_annotations[k]['tags'],
                actual_annotations[k]['tags']
            )
            self.assertEqual(
                expected_annotations[k]['times'][0],
                actual_annotations[k]['times'][0]
            )
            self.assertEqual(
                expected_annotations[k]['freqs'][0],
                actual_annotations[k]['freqs'][0]
            )

        self.assertEqual(expected_annot_activity[2], actual_annot_activity[2])
        self.assertEqual(
            expected_annot_activity[4][0], actual_annot_activity[4][0])
        self.assertEqual(
            expected_annot_activity[5][0], actual_annot_activity[5][0])
        self.assertEqual(expected_indices, actual_indices)


class TestSaveAnnotation(unittest.TestCase):

    def test_empty(self):
        actual = CMD.save_annotation([], [], 'data/save_test.csv')
        expected = None
        self.assertEqual(expected, actual)

    def test_nonempty(self):
        actual = CMD.save_annotation(
            [0.5, 0.5, 0.5, 1.5], [200.0, 0.0, 201, 400],
            'data/save_test.csv')
        expected = 'data/save_test.csv'
        self.assertEqual(expected, actual)
        self.assertTrue(os.path.exists(expected))
        os.remove(expected)


class TestCreateAnnotations(unittest.TestCase):

    def test_one(self):
        save_dir = 'asdf/fdasa'
        track_id = 'Artist_Title'
        stem_annotations = {
            2: {
                'tags': ['multif0', 'bass'],
                'times': [0.0, 0.1, 0.2],
                'freqs': [100, 100, 100]
            },
            3: {
                'tags': ['multif0', 'piano'],
                'times': [0.0, 0.1, 0.2],
                'freqs': [200, 200, 200]
            },
            7: {
                'tags': ['multif0', 'guitar'],
                'times': [1.0, 1.0, 1.1, 1.2],
                'freqs': [100, 200, 100, 100]
            },
            9: {
                'tags': ['multif0', 'melody'],
                'times': [0.2, 0.3, 0.4],
                'freqs': [100, 100, 100]
            },
            1: {
                'tags': ['multif0', 'vocal'],
                'times': [1.0, 1.1, 1.2],
                'freqs': [400, 400, 400]
            }
        }

        actual = CMD.create_annotations(save_dir, track_id, stem_annotations)
        expected = {
            'multif0': {
                'times': [
                    1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2,
                    1.0, 1.0, 1.1, 1.2, 0.2, 0.3, 0.4],
                'freqs': [
                    400, 400, 400, 100, 100, 100, 200, 200, 200,
                    100, 200, 100, 100, 100, 100, 100],
                'path': "asdf/fdasa/Artist_Title_multif0_annotation.txt"
            },
            'multif0_noguitar': {
                'times': [
                    1.0, 1.1, 1.2, 0.0, 0.1, 0.2,
                    0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
                'freqs': [
                    400, 400, 400, 100, 100, 100,
                    200, 200, 200, 100, 100, 100],
                'path': 
                    "asdf/fdasa/Artist_Title_multif0_noguitar_annotation.txt"
            },
            'multif0_nosynth': {
                'times': [1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
                'freqs': [400, 400, 400, 100, 100, 100, 100, 100, 100],
                'path': "asdf/fdasa/Artist_Title_multif0_nosynth_annotation.txt"
            },
            'melody': {
                'times': [0.2, 0.3, 0.4],
                'freqs': [100, 100, 100],
                'path': "asdf/fdasa/Artist_Title_melody_annotation.txt"
            },
            'vocal': {
                'times': [1.0, 1.1, 1.2],
                'freqs': [400, 400, 400],
                'path': "asdf/fdasa/Artist_Title_vocal_annotation.txt"
            },
            'bass': {
                'times': [0.0, 0.1, 0.2],
                'freqs': [100, 100, 100],
                'path': "asdf/fdasa/Artist_Title_bass_annotation.txt"
            },
            'piano': {
                'times': [0.0, 0.1, 0.2],
                'freqs': [200, 200, 200],
                'path': "asdf/fdasa/Artist_Title_piano_annotation.txt"
            },
            'guitar': {
                'times': [1.0, 1.0, 1.1, 1.2],
                'freqs': [100, 200, 100, 100],
                'path': "asdf/fdasa/Artist_Title_guitar_annotation.txt"
            }
        }
        self.assertEqual(sorted(expected.keys()), sorted(actual.keys()))
        for key in actual.keys():
            self.assertEqual(expected[key], actual[key])


class TestCreateAnnotationSavePairs(unittest.TestCase):

    def test_one(self):
        annotations = {
            'multif0': {
                'times': [
                    1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2,
                    1.0, 1.0, 1.1, 1.2, 0.2, 0.3, 0.4],
                'freqs': [
                    400, 400, 400, 100, 100, 100, 200, 200, 200,
                    100, 200, 100, 100, 100, 100, 100],
                'path': "data/Artist_Title_multif0_annotation.txt"
            },
            'multif0_noguitar': {
                'times': [
                    1.0, 1.1, 1.2, 0.0, 0.1, 0.2,
                    0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
                'freqs': [
                    400, 400, 400, 100, 100, 100,
                    200, 200, 200, 100, 100, 100],
                'path': 
                    "data/Artist_Title_multif0_noguitar_annotation.txt"
            },
            'multif0_nosynth': {
                'times': [1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
                'freqs': [400, 400, 400, 100, 100, 100, 100, 100, 100],
                'path': "data/Artist_Title_multif0_nosynth_annotation.txt"
            },
            'melody': {
                'times': [0.2, 0.3, 0.4],
                'freqs': [100, 100, 100],
                'path': "data/Artist_Title_melody_annotation.txt"
            },
            'vocal': {
                'times': [1.0, 1.1, 1.2],
                'freqs': [400, 400, 400],
                'path': "data/Artist_Title_vocal_annotation.txt"
            },
            'bass': {
                'times': [0.0, 0.1, 0.2],
                'freqs': [100, 100, 100],
                'path': "data/Artist_Title_bass_annotation.txt"
            },
            'piano': {
                'times': [0.0, 0.1, 0.2],
                'freqs': [200, 200, 200],
                'path': "data/Artist_Title_piano_annotation.txt"
            },
            'guitar': {
                'times': [1.0, 1.0, 1.1, 1.2],
                'freqs': [100, 200, 100, 100],
                'path': "data/Artist_Title_guitar_annotation.txt"
            }
        }

        mix_path = 'Fakemix.wav'
        mix_path_noguitar = 'Fakemix_noguitar.wav'
        mix_path_nosynth = 'Fakemix_nosynth.wav'
        actual = CMD.create_annotation_save_pairs(
            annotations, mix_path, mix_path_noguitar,
            mix_path_nosynth
        )

        expected = {
            'Fakemix.wav': {
                'multif0': "data/Artist_Title_multif0_annotation.txt",
                'guitar': "data/Artist_Title_guitar_annotation.txt",
                'piano': "data/Artist_Title_piano_annotation.txt",
                'vocal': "data/Artist_Title_vocal_annotation.txt",
                'melody': "data/Artist_Title_melody_annotation.txt",
                'bass': "data/Artist_Title_bass_annotation.txt"
            },
            'Fakemix_noguitar.wav': {
                'multif0': "data/Artist_Title_multif0_noguitar_annotation.txt",
                'guitar': None,
                'piano': "data/Artist_Title_piano_annotation.txt",
                'vocal': "data/Artist_Title_vocal_annotation.txt",
                'melody': "data/Artist_Title_melody_annotation.txt",
                'bass': "data/Artist_Title_bass_annotation.txt"
            },
            'Fakemix_nosynth.wav': {
                'multif0': "data/Artist_Title_multif0_nosynth_annotation.txt",
                'guitar': None,
                'piano': None,
                'vocal': "data/Artist_Title_vocal_annotation.txt",
                'melody': "data/Artist_Title_melody_annotation.txt",
                'bass': "data/Artist_Title_bass_annotation.txt"
            },
        }

        self.assertEqual(sorted(expected.keys()), sorted(actual.keys()))
        files = []
        for key in actual.keys():
            self.assertEqual(
                sorted(expected[key].keys()), sorted(actual[key].keys()))
            for subkey in actual[key].keys():
                self.assertEqual(expected[key][subkey], actual[key][subkey])
                if actual[key][subkey] is not None:
                    self.assertTrue(os.path.exists(actual[key][subkey]))
                    files.append(actual[key][subkey])
        files = list(set(files))
        for f in files:
            os.remove(f)

    def test_two(self):
        annotations = {
            'multif0': {
                'times': [
                    1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2,
                    1.0, 1.0, 1.1, 1.2, 0.2, 0.3, 0.4],
                'freqs': [
                    400, 400, 400, 100, 100, 100, 200, 200, 200,
                    100, 200, 100, 100, 100, 100, 100],
                'path': "data/Artist_Title_multif0_annotation.txt"
            },
            'multif0_noguitar': {
                'times': [
                    1.0, 1.1, 1.2, 0.0, 0.1, 0.2,
                    0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
                'freqs': [
                    400, 400, 400, 100, 100, 100,
                    200, 200, 200, 100, 100, 100],
                'path': 
                    "data/Artist_Title_multif0_noguitar_annotation.txt"
            },
            'multif0_nosynth': {
                'times': [1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
                'freqs': [400, 400, 400, 100, 100, 100, 100, 100, 100],
                'path': "data/Artist_Title_multif0_nosynth_annotation.txt"
            },
            'melody': {
                'times': [0.2, 0.3, 0.4],
                'freqs': [100, 100, 100],
                'path': "data/Artist_Title_melody_annotation.txt"
            },
            'vocal': {
                'times': [1.0, 1.1, 1.2],
                'freqs': [400, 400, 400],
                'path': "data/Artist_Title_vocal_annotation.txt"
            },
            'piano': {
                'times': [0.0, 0.1, 0.2],
                'freqs': [200, 200, 200],
                'path': "data/Artist_Title_piano_annotation.txt"
            },
            'guitar': {
                'times': [1.0, 1.0, 1.1, 1.2],
                'freqs': [100, 200, 100, 100],
                'path': "data/Artist_Title_guitar_annotation.txt"
            }
        }

        mix_path = 'Fakemix.wav'
        mix_path_noguitar = 'Fakemix_noguitar.wav'
        mix_path_nosynth = 'Fakemix_nosynth.wav'
        actual = CMD.create_annotation_save_pairs(
            annotations, mix_path, mix_path_noguitar,
            mix_path_nosynth
        )

        expected = {
            'Fakemix.wav': {
                'multif0': "data/Artist_Title_multif0_annotation.txt",
                'guitar': "data/Artist_Title_guitar_annotation.txt",
                'piano': "data/Artist_Title_piano_annotation.txt",
                'vocal': "data/Artist_Title_vocal_annotation.txt",
                'melody': "data/Artist_Title_melody_annotation.txt"
            },
            'Fakemix_noguitar.wav': {
                'multif0': "data/Artist_Title_multif0_noguitar_annotation.txt",
                'guitar': None,
                'piano': "data/Artist_Title_piano_annotation.txt",
                'vocal': "data/Artist_Title_vocal_annotation.txt",
                'melody': "data/Artist_Title_melody_annotation.txt"
            },
            'Fakemix_nosynth.wav': {
                'multif0': "data/Artist_Title_multif0_nosynth_annotation.txt",
                'guitar': None,
                'piano': None,
                'vocal': "data/Artist_Title_vocal_annotation.txt",
                'melody': "data/Artist_Title_melody_annotation.txt"
            }
        }

        self.assertEqual(sorted(expected.keys()), sorted(actual.keys()))
        files = []
        for key in actual.keys():
            self.assertEqual(
                sorted(expected[key].keys()), sorted(actual[key].keys()))
            for subkey in actual[key].keys():
                self.assertEqual(expected[key][subkey], actual[key][subkey])
                if actual[key][subkey] is not None:
                    self.assertTrue(os.path.exists(actual[key][subkey]))
                    files.append(actual[key][subkey])
        files = list(set(files))
        for f in files:
            os.remove(f)


class TestGenerateFilteredStems(unittest.TestCase):

    def test_empty(self):
        stem_annot_activity = {}
        mtrack = mdb.MultiTrack("MusicDelta_BebopJazz")
        save_dir = ""
        actual = CMD.generate_filtered_stems(
            stem_annot_activity, mtrack, save_dir)
        expected = {}
        self.assertEqual(expected, actual)

    def test_nonempty_one(self):
        stem_annot_activity = {1: None, 2: np.ones((20*44100, ))}
        mtrack = mdb.MultiTrack("MusicDelta_Reggae")
        save_dir = 'data'
        actual = CMD.generate_filtered_stems(
            stem_annot_activity, mtrack, save_dir)
        expected = {2: 'data/MusicDelta_Reggae_STEM_2_alt.wav'}
        self.assertEqual(expected, actual)
        self.assertTrue(os.path.exists('data/MusicDelta_Reggae_STEM_2_alt.wav'))
        os.remove('data/MusicDelta_Reggae_STEM_2_alt.wav')

    def test_nonempty_two(self):
        stem_annot_activity = {1: None, 2: np.ones((2*44100, ))}
        mtrack = mdb.MultiTrack("MusicDelta_Reggae")
        save_dir = 'data'
        actual = CMD.generate_filtered_stems(
            stem_annot_activity, mtrack, save_dir)
        expected = {2: 'data/MusicDelta_Reggae_STEM_2_alt.wav'}
        self.assertEqual(expected, actual)
        self.assertTrue(os.path.exists('data/MusicDelta_Reggae_STEM_2_alt.wav'))
        os.remove('data/MusicDelta_Reggae_STEM_2_alt.wav')


class TestCreateMixes(unittest.TestCase):

    def test_one(self):
        mtrack = mdb.MultiTrack("AHa_TakeOnMe")
        mix_path = 'data/test_mix.wav'
        mix_path_noguitar = 'data/test_mix_noguitar.wav'
        mix_path_nosynth = 'data/test_mix_nosynth.wav'
        stem_indices = [1, 2, 3, 4, 5, 6]
        stem_indices_guitar = [4]
        stem_indices_piano = [1, 3, 6]
        resynth_path = '/Users/rabitt/Dropbox/piano_guitar_resynth/resynth'
        replace_path = os.path.join(
            '/Users/rabitt/Dropbox',
            'piano_guitar_resynth/vocalists_replace'
        )
        altfiles = {
            1: os.path.join(resynth_path, "AHa_TakeOnMe_STEM_01_resynth.wav"),
            2: os.path.join(replace_path, "AHa_TakeOnMe_STEM_02_replace.wav"),
            3: os.path.join(resynth_path, "AHa_TakeOnMe_STEM_03_resynth.wav"),
            4: os.path.join(resynth_path, "AHa_TakeOnMe_STEM_04_resynth.wav"),
            6: os.path.join(resynth_path, "AHa_TakeOnMe_STEM_06_resynth.wav")
        }
        (actual_mix,
         actual_mix_noguitar,
         actual_mix_nosynth) = CMD.create_mixes(
             mtrack, mix_path, mix_path_noguitar, mix_path_nosynth,
             stem_indices, stem_indices_guitar, stem_indices_piano,
             altfiles
         )
        expected_mix = sorted([
            os.path.join(resynth_path, "AHa_TakeOnMe_STEM_01_resynth.wav"),
            os.path.join(replace_path, "AHa_TakeOnMe_STEM_02_replace.wav"),
            os.path.join(resynth_path, "AHa_TakeOnMe_STEM_03_resynth.wav"),
            os.path.join(resynth_path, "AHa_TakeOnMe_STEM_04_resynth.wav"),
            os.path.join(
                mdb.MEDLEYDB_PATH, 'Audio', 'AHa_TakeOnMe',
                'AHa_TakeOnMe_STEMS', "AHa_TakeOnMe_STEM_05.wav"),
            os.path.join(resynth_path, "AHa_TakeOnMe_STEM_06_resynth.wav")
        ])
        expected_mix_noguitar = sorted([
            os.path.join(resynth_path, "AHa_TakeOnMe_STEM_01_resynth.wav"),
            os.path.join(replace_path, "AHa_TakeOnMe_STEM_02_replace.wav"),
            os.path.join(resynth_path, "AHa_TakeOnMe_STEM_03_resynth.wav"),
            os.path.join(
                mdb.MEDLEYDB_PATH, 'Audio', 'AHa_TakeOnMe',
                'AHa_TakeOnMe_STEMS', "AHa_TakeOnMe_STEM_05.wav"),
            os.path.join(resynth_path, "AHa_TakeOnMe_STEM_06_resynth.wav")
        ])
        expected_mix_nosynth = sorted([
            os.path.join(replace_path, "AHa_TakeOnMe_STEM_02_replace.wav"),
            os.path.join(
                mdb.MEDLEYDB_PATH, 'Audio', 'AHa_TakeOnMe',
                'AHa_TakeOnMe_STEMS', "AHa_TakeOnMe_STEM_05.wav"),
        ])

        self.assertEqual(expected_mix, sorted(actual_mix))
        self.assertEqual(expected_mix_noguitar, sorted(actual_mix_noguitar))
        self.assertEqual(expected_mix_nosynth, sorted(actual_mix_nosynth))

        self.assertTrue(os.path.exists(mix_path))
        os.remove(mix_path)
        self.assertTrue(os.path.exists(mix_path_noguitar))
        os.remove(mix_path_noguitar)
        self.assertTrue(os.path.exists(mix_path_nosynth))
        os.remove(mix_path_nosynth)


class TestCreateCompleteResynthMix(unittest.TestCase):

    def test_one(self):
        mtrack = mdb.MultiTrack("AHa_TakeOnMe")
        resynth_path = '/Users/rabitt/Dropbox/piano_guitar_resynth/resynth'
        replace_path = os.path.join(
            '/Users/rabitt/Dropbox',
            'piano_guitar_resynth/vocalists_replace'
        )
        save_dir = 'data'
        actual = CMD.create_complete_resynth_mix(
            mtrack, resynth_path, replace_path, save_dir)
        expected = {
            "data/AHa_TakeOnMe_MIX_complete_resynth.wav": {
                'multif0': "data/AHa_TakeOnMe_multif0_annotation.txt",
                'guitar': "data/AHa_TakeOnMe_guitar_annotation.txt",
                'piano': None,
                'vocal': "data/AHa_TakeOnMe_vocal_annotation.txt",
                'melody': "data/AHa_TakeOnMe_melody_annotation.txt",
                'bass': None
            },
            "data/AHa_TakeOnMe_MIX_complete_noguitar.wav": {
                'multif0': "data/AHa_TakeOnMe_multif0_noguitar_annotation.txt",
                'guitar': None,
                'piano': None,
                'vocal': "data/AHa_TakeOnMe_vocal_annotation.txt",
                'melody': "data/AHa_TakeOnMe_melody_annotation.txt",
                'bass': None
            },
            "data/AHa_TakeOnMe_MIX_complete_nosynth.wav": {
                'multif0': "data/AHa_TakeOnMe_multif0_nosynth_annotation.txt",
                'guitar': None,
                'piano': None,
                'vocal': "data/AHa_TakeOnMe_vocal_annotation.txt",
                'melody': "data/AHa_TakeOnMe_melody_annotation.txt",
                'bass': None
            }
        }

        self.assertEqual(
            sorted(expected.keys()),
            sorted(actual.keys())
        )
        files = []
        for key in actual.keys():
            self.assertEqual(
                sorted(expected[key].keys()),
                sorted(actual[key].keys()))
            self.assertTrue(os.path.exists(key))
            os.remove(key)
            for subkey in actual[key].keys():
                if actual[key][subkey] is not None:
                    self.assertEqual(
                        expected[key][subkey],
                        actual[key][subkey]
                    )
                    self.assertTrue(os.path.exists(actual[key][subkey]))
                    files.append(actual[key][subkey])
                else:
                    self.assertEqual(expected[key][subkey], actual[key][subkey])

        for fpath in list(set(files)):
            os.remove(fpath)


class TestGetAnnotationMono(unittest.TestCase):

    def test_empty_list(self):
        mtrack = mdb.MultiTrack("MusicDelta_BebopJazz")
        stem_idx = []
        stem_list = [mtrack.stems[s] for s in stem_idx]
        actual_times, actual_freqs = CMD.get_annotation_mono(mtrack, stem_list)
        expected_times = []
        expected_freqs = []
        self.assertEqual(expected_times, actual_times)
        self.assertEqual(expected_freqs, actual_freqs)

    def test_nonempty_list_one(self):
        mtrack = mdb.MultiTrack("MusicDelta_BebopJazz")
        stem_idx = [4, 5]
        stem_list = [mtrack.stems[s] for s in stem_idx]
        actual_times, actual_freqs = CMD.get_annotation_mono(mtrack, stem_list)
        expected_times = [0.39473922900000002]
        expected_freqs = [193.88999999999999]
        self.assertEqual(expected_times[0], actual_times[0])
        self.assertEqual(expected_freqs[0], actual_freqs[0])

    def test_nonempty_list_two(self):
        mtrack = mdb.MultiTrack("MusicDelta_BebopJazz")
        stem_idx = [3, 4, 5]
        stem_list = [mtrack.stems[s] for s in stem_idx]
        actual_times, actual_freqs = CMD.get_annotation_mono(mtrack, stem_list)
        expected_times = None
        expected_freqs = None
        self.assertEqual(expected_times, actual_times)
        self.assertEqual(expected_freqs, actual_freqs)


class TestGetFullmixAnnotation(unittest.TestCase):

    def test_one(self):
        mtrack = mdb.MultiTrack("AHa_TakeOnMe")
        save_dir = 'data'
        actual = CMD.get_fullmix_annotations(mtrack, save_dir)
        expected = {mtrack.mix_path: {}}
        self.assertEqual(expected, actual)

    def test_two(self):
        mtrack = mdb.MultiTrack("AcDc_BackInBlack")
        save_dir = 'data'
        actual = CMD.get_fullmix_annotations(mtrack, save_dir)
        expected = {
            mtrack.mix_path: {
                'vocal': 'data/AcDc_BackInBlack_MIX_vocal.txt',
                'bass': 'data/AcDc_BackInBlack_MIX_bass.txt'
            }
        }

        self.assertEqual(
            sorted(expected.keys()),
            sorted(actual.keys())
        )
        files = []
        for key in actual.keys():
            self.assertEqual(
                sorted(expected[key].keys()),
                sorted(actual[key].keys()))
            for subkey in actual[key].keys():
                if actual[key][subkey] is not None:
                    self.assertEqual(
                        expected[key][subkey],
                        actual[key][subkey]
                    )
                    self.assertTrue(os.path.exists(actual[key][subkey]))
                    files.append(actual[key][subkey])
                else:
                    self.assertEqual(expected[key][subkey], actual[key][subkey])

        for fpath in list(set(files)):
            os.remove(fpath)

    def test_three(self):
        mtrack = mdb.MultiTrack("MusicDelta_BebopJazz")
        save_dir = 'data'
        actual = CMD.get_fullmix_annotations(mtrack, save_dir)
        expected = {
            mtrack.mix_path: {
                'vocal': None,
                'bass': 'data/MusicDelta_BebopJazz_MIX_bass.txt',
                'melody': 'data/MusicDelta_BebopJazz_MIX_melody.txt'
            }
        }

        self.assertEqual(
            sorted(expected.keys()),
            sorted(actual.keys())
        )
        files = []
        for key in actual.keys():
            self.assertEqual(
                sorted(expected[key].keys()),
                sorted(actual[key].keys()))
            for subkey in actual[key].keys():
                if actual[key][subkey] is not None:
                    self.assertEqual(
                        expected[key][subkey],
                        actual[key][subkey]
                    )
                    self.assertTrue(os.path.exists(actual[key][subkey]))
                    files.append(actual[key][subkey])
                else:
                    self.assertEqual(expected[key][subkey], actual[key][subkey])

        for fpath in list(set(files)):
            os.remove(fpath)

    def test_four(self):
        mtrack = mdb.MultiTrack("TablaBreakbeatScience_Animoog")
        save_dir = 'data'
        actual = CMD.get_fullmix_annotations(mtrack, save_dir)
        expected = {
            mtrack.mix_path: {
                'vocal': None,
                'bass': None,
                'melody': None
            }
        }

        self.assertEqual(
            sorted(expected.keys()),
            sorted(actual.keys())
        )
        files = []
        for key in actual.keys():
            self.assertEqual(
                sorted(expected[key].keys()),
                sorted(actual[key].keys()))
            for subkey in actual[key].keys():
                if actual[key][subkey] is not None:
                    self.assertEqual(
                        expected[key][subkey],
                        actual[key][subkey]
                    )
                    self.assertTrue(os.path.exists(actual[key][subkey]))
                    files.append(actual[key][subkey])
                else:
                    self.assertEqual(expected[key][subkey], actual[key][subkey])

        for fpath in list(set(files)):
            os.remove(fpath)


class TestGetAllAudioAnnotPairs(unittest.TestCase):

    def test_one(self):
        mtrack = mdb.MultiTrack('AHa_TakeOnMe')
        save_dir = 'data'
        resynth_path = '/Users/rabitt/Dropbox/piano_guitar_resynth/resynth'
        replace_path = os.path.join(
            '/Users/rabitt/Dropbox',
            'piano_guitar_resynth/vocalists_replace'
        )
        actual = CMD.get_all_audio_annot_pairs(
            mtrack, save_dir, resynth_path, replace_path)

        expected = 'data/AHa_TakeOnMe_training_pairs.json'
        self.assertEqual(expected, actual)
        self.assertTrue(os.path.exists(actual))


