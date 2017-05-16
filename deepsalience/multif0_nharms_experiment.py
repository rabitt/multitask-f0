from __future__ import print_function
import argparse
import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os

import experiment_datasets

def model_def(n_harms):
    ### DEFINE MODEL ###
    input_shape = (None, None, n_harms)
    inputs = Input(shape=input_shape)

    y0 = BatchNormalization()(inputs)
    y1 = Conv2D(128, (5, 5), padding='same', activation='relu', name='bendy1')(y0)
    y1a = BatchNormalization()(y1)
    y2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='bendy2')(y1a)
    y2a = BatchNormalization()(y2)
    y3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy1')(y2a)
    y3a = BatchNormalization()(y3)
    y4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy2')(y3a)
    y4a = BatchNormalization()(y4)
    y5 = Conv2D(8, (70, 3), padding='same', activation='relu', name='distribute')(y4a)
    y5a = BatchNormalization()(y5)
    y6 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y5a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y6)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def main(args):
    n_harms = args.n_harms
    save_key = "{}_{}_harmonics".format(
        os.path.basename(__file__).split('.')[0], n_harms)
    
    model = model_def(n_harms)
    if n_harms == 6:
        experiment_datasets.experiment(
            save_key, model, 'multif0_complete', n_harms=None)
    else:
        experiment_datasets.experiment(
            save_key, model, 'multif0_complete', n_harms=n_harms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment for testing the number of harmonics")
    parser.add_argument("n_harms",
                        type=int,
                        help="Number of harmonics to use in computation")
    main(parser.parse_args())
