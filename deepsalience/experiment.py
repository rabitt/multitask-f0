"""common code for all experiments
"""
from __future__ import print_function

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import numpy as np
np.random.seed(1337)
import keras
import medleydb as mdb
import os

import core
import evaluate

SAMPLES_PER_EPOCH = 512
NB_EPOCHS = 100
NB_VAL_SAMPLES = 512


def train(model, model_save_path):
    
    data_path = core.data_path_multif0_complete()
    mtrack_list = core.track_id_list()
    input_patch_size = core.patch_size()

    ### DATA SETUP ###
    dat = core.Data(
        mtrack_list, data_path, input_patch_size=input_patch_size
    )
    train_generator = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()

    model.compile(
        loss=core.bkld, metrics=['mse', core.soft_binary_accuracy],
        optimizer='adam'
    )

    print(model.summary(line_length=80))

    ### FIT MODEL ###
    history = model.fit_generator(
        train_generator, SAMPLES_PER_EPOCH, epochs=NB_EPOCHS, verbose=1,
        validation_data=validation_generator, validation_steps=NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_save_path, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=15, verbose=0)
        ]
    )

    ### load best weights ###
    model.load_weights(model_save_path)

    return model, history, dat


def evaluate(exper_dir, save_key, history, dat, model):

    (save_path, _, plot_save_path,
     model_scores_path, scores_path, score_summary_path
    ) = evaluate.get_paths(exper_dir, save_key)

    ### Results plots ###
    print("plotting results...")
    evaluate.plot_metrics_epochs(history, plot_save_path)

    ### Evaluate ###
    print("getting model metrics...")
    evaluate.get_model_metrics(dat, model, model_scores_path)

    print("getting multif0 metrics...")
    evaluate.get_all_multif0_metrics(
        dat.test_files, model, save_path, scores_path, score_summary_path
    )

    bach10_files = core.get_file_paths(mdb.TRACK_LIST_BACH10, dat.data_path)
    evaluate.get_all_multif0_metrics(
        bach10_files, model,
        save_path,
        os.path.join(save_path, "bach10_scores.csv"),
        os.path.join(save_path, "bach10_score_summary.csv"), create_pred=True)


def experiment(save_key, model):
    """common code for all experiments
    """
    exper_dir = core.experiment_output_path()

    (save_path, model_save_path, _,_, _, _) = evaluate.get_paths(
        exper_dir, save_key
    )

    model, history, dat = train(model, model_save_path)

    evaluate(exper_dir, save_key, history, dat, model)

    print("done!")
    print("Results saved to {}".format(save_path))


