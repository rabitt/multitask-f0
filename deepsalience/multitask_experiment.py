import multitask_core as MC
import multitask_evaluate as ME
import keras

def main(args):
    data_splits = MC.load_data_splits()

    if args.data_types is None:
        data_types = MC.DATA_TYPES
    else:
        data_types = args.data_types

    if args.tasks is None:
        tasks = MC.TASKS
    else:
        tasks = args.tasks

    if args.mux_weights is None:
        mux_weights = None
    else:
        mux_weights = args.mux_weights

    train_generator = MC.multitask_generator(
        data_splits['train'], data_types=data_types,
        tasks=tasks, mux_weights=mux_weights)
    validate_generator = MC.multitask_generator(
        data_splits['validate'], data_types=data_types,
        tasks=tasks, mux_weights=mux_weights)
    test_generator = MC.multitask_generator(
        data_splits['test'], data_types=data_types,
        tasks=tasks, mux_weights=mux_weights)

    loss_weights = args.loss_weights
    sample_weight_mode = args.sample_weight_mode
    model_save_path = args.model_save_path

    model = args.model
    model.compile(
        loss=MC.bkld, metrics=['mse', MC.soft_binary_accuracy],
        loss_weights=loss_weights,
        optimizer='adam', sample_weight_mode=sample_weight_mode
    )

    SAMPLES_PER_EPOCH = 50
    NB_EPOCHS = 200
    NB_VAL_SAMPLES = 10

    history = model.fit_generator(
        train_generator, SAMPLES_PER_EPOCH, epochs=NB_EPOCHS, verbose=1,
        validation_data=validate_generator, validation_steps=NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_save_path, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=25, verbose=0)
        ]
    )

    #TODO FIGURE OUT OUTPUT PATHS GARBAGE

    MC.history_plot(history, tasks, args.history_plot_path)

    model.load_weights(model_save_path)

    thresholds, scores = ME.evaluate_model(model, tasks)
    ME.save_eval(args.eval_output_path, thresholds, scores)

