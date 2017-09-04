import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multitask_experiment
import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda, Permute
from keras.layers.merge import Concatenate, Multiply
from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def get_model():
    input_shape = (None, None, 5)
    y0 = Input(shape=input_shape, name='input')
    y_freq = Input(shape=(None, None, 1), name='freq_map')

    y1_pitch = Conv2D(
        32, (5, 5), padding='same', activation='relu', name='pitch_layer1')(y0)
    y1a_pitch = BatchNormalization()(y1_pitch)
    y2_pitch = Conv2D(
        32, (5, 5), padding='same', activation='relu', name='pitch_layer2')(y1a_pitch)
    y2a_pitch = BatchNormalization()(y2_pitch)
    y3_pitch = Conv2D(32, (3, 3), padding='same', activation='relu', name='smoothy2')(y2a_pitch)
    y3a_pitch = BatchNormalization()(y3_pitch)
    y4_pitch = Conv2D(8, (70, 3), padding='same', activation='relu', name='distribute')(y3a_pitch)
    y4a_pitch = BatchNormalization()(y4_pitch)

    y_multif0 = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='multif0_presqueeze')(y4a_pitch)
    multif0 = Lambda(lambda x: K.squeeze(x, axis=3), name='multif0')(y_multif0)

    y_mask = Multiply(name='mask')([y_multif0, y0])
    y1_timbre = Conv2D(
        512, (2, 3), padding='same', activation='relu', name='timbre_layer1')(y_mask)
    y1a_timbre = BatchNormalization()(y1_timbre)

    y_concat = Concatenate(name='timbre_and_pitch')([y_multif0, y1a_timbre])
    ya_concat = BatchNormalization()(y_concat)
    y_concat_freq = Concatenate(name='freq_concat')([ya_concat, y_freq])

    y_mel_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='melody_filters')(y_concat_freq)
    ya_mel_feat = BatchNormalization()(y_mel_feat)
    y_mel_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='melody_filters2')(ya_mel_feat)
    ya_mel_feat2 = BatchNormalization()(y_mel_feat2)

    y_mel_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='melody_filters4')(ya_mel_feat2)
    ya_mel_feat4 = BatchNormalization()(y_mel_feat4)
    y_mel_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='melody_filters5')(ya_mel_feat4)
    ya_mel_feat5 = BatchNormalization()(y_mel_feat5)

    y_bass_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='bass_filters')(y_concat_freq)
    ya_bass_feat = BatchNormalization()(y_bass_feat)
    y_bass_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='bass_filters2')(ya_bass_feat)
    ya_bass_feat2 = BatchNormalization()(y_bass_feat2)

    y_bass_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='bass_filters4')(ya_bass_feat2)
    ya_bass_feat4 = BatchNormalization()(y_bass_feat4)
    y_bass_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='bass_filters5')(ya_bass_feat4)
    ya_bass_feat5 = BatchNormalization()(y_bass_feat5)

    y_vocal_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='vocal_filters')(y_concat_freq)
    ya_vocal_feat = BatchNormalization()(y_vocal_feat)
    y_vocal_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='vocal_filters2')(ya_vocal_feat)
    ya_vocal_feat2 = BatchNormalization()(y_vocal_feat2)

    y_vocal_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='vocal_filters4')(ya_vocal_feat2)
    ya_vocal_feat4 = BatchNormalization()(y_vocal_feat4)
    y_vocal_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='vocal_filters5')(ya_vocal_feat4)
    ya_vocal_feat5 = BatchNormalization()(y_vocal_feat5)

    y_melody = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='melody_presqueeze')(ya_mel_feat5)
    melody = Lambda(lambda x: K.squeeze(x, axis=3), name='melody')(y_melody)

    y_bass = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='bass_presqueeze')(ya_bass_feat5)
    bass = Lambda(lambda x: K.squeeze(x, axis=3), name='bass')(y_bass)

    y_vocal = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='vocal_presqueeze')(ya_vocal_feat5)
    vocal = Lambda(lambda x: K.squeeze(x, axis=3), name='vocal')(y_vocal)

    model = Model(inputs=[y0, y_freq], outputs=[multif0, melody, bass, vocal])

    model.summary(line_length=120)

    return model


model = get_model()
output_path = '../../experiment_output/multitask_freq_nosynthdata'
tasks = None
data_types = ['XA', 'XD']
loss_weights = {'multif0': 2.0, 'melody': 1.0, 'bass': 1.0, 'vocal': 1.0}
sample_weight_mode = {'multif0': None, 'melody': None, 'bass': None, 'vocal': None}
task_indices = {'multif0': 0, 'melody': 1, 'bass': 2, 'vocal': 3}

multitask_experiment.main(
    model, output_path, loss_weights, sample_weight_mode,
    task_indices, data_types=data_types, tasks=tasks, mux_weights=None,
    samples_per_epoch=50, nb_epochs=200, nb_val_samples=50,
    freq_feature=True
)
