import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path
import argparse
import shutil
import sys
import os

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.absolute()))

from models import lenet5

def preprocess_input(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x -= 0.5
    y = tf.one_hot(y, depth=10)
    return x, y


def list_of_floats(arg):
    return list(map(float, arg.split(',')))


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def list_of_tuples(arg):
    tuples = arg.split(';')
    return [tuple(map(int, tup.split(','))) for tup in tuples]

def broadcast_to_length(sequence, length):
    return sequence * length


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default='fashion_mnist/base-relu-lenet5', required=False)
    parser.add_argument('--seed', type=int, default=3, required=False)
    parser.add_argument('--batch-size', type=int, default=32, required=False)
    parser.add_argument('--learning-rate', type=float, default=0.0001, required=False)
    parser.add_argument('--epochs', type=int, default=100, required=False)
    parser.add_argument('--steps-per-epoch', type=int, default=1000, required=False)
    args = parser.parse_args()

    output_path = Path(args.output_path)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path)
    
    seed = args.seed
    tf.keras.utils.set_random_seed(seed)

    batch_size = args.batch_size
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    validation_steps = None
    test_steps = None

    input_shape = (28, 28, 1)
    n_classes = 10

    model = lenet5(n_classes, input_shape)
    model.summary()

    lr = args.learning_rate
    warmup_steps = steps_per_epoch * 2
    decay_steps = steps_per_epoch * 8
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr, decay_steps=decay_steps, alpha=lr/10.0,
                                                            warmup_target=lr*10.0, warmup_steps=warmup_steps)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    ds_train, ds_test = tfds.load('fashion_mnist',
                                  split=['train', 'test'],
                                  shuffle_files=True,
                                  as_supervised=True,
                                  with_info=False,
                                  download=False) # Change this if necessary

    ds_train = ds_train.cache().shuffle(1000).batch(batch_size).map(preprocess_input).repeat().prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(500).map(preprocess_input).cache().prefetch(tf.data.AUTOTUNE)

    checkpoints_path = output_path / 'model'
    os.makedirs(checkpoints_path)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path / 'weights.h5', monitor='val_categorical_accuracy',
                                                              save_weights_only=True, save_best_only=True, mode='max')
    
    log_dir = output_path / 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [checkpoints_callback, tensorboard_callback]

    history = model.fit(x=ds_train, validation_data=ds_test, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps, callbacks=callbacks)

    history_dict = history.history
    loss_list = history_dict['val_loss']
    acc_list = history_dict['val_categorical_accuracy']
    best_acc_idx = acc_list.index(max(acc_list))
    
    tf.keras.models.save_model(model, output_path / 'keras_model.h5', save_format='h5')

    print('\n')
    print('Test loss:', loss_list[best_acc_idx])
    print('Test accuracy:', acc_list[best_acc_idx])

# Test acc: 89.31%
