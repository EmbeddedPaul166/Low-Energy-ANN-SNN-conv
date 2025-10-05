import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path
import argparse
import shutil
import sys
import os

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.absolute()))

from models import vgg16

INPUT_SHAPE = [48, 48]

rotate = tf.keras.layers.RandomRotation(0.2, 'reflect', 'bilinear')
shift = tf.keras.layers.RandomTranslation(0.2, 0.2, 'reflect', 'bilinear')


def preprocess_input(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x -= 0.5
    x = tf.image.resize(x, INPUT_SHAPE, 'bicubic')
    y = tf.one_hot(y, depth=100)
    return x, y


@tf.function
def augment(x, y, bs):
    """
    Augmentations in order:
    1. Flip left-right
    2. Brightness
    3. Contrast
    4. Rotation
    5. Shift
    """
    rn_shape = (bs, 1, 1, 1)

    p_flip = tf.random.uniform(rn_shape, 0.0, 1.0)
    x = tf.where(p_flip > 0.5,
                 tf.image.flip_left_right(x),
                 x)
    
    p_brightness = tf.random.uniform(rn_shape, 0.0, 1.0)
    delta = tf.random.uniform(rn_shape, -0.2, 0.2)
    x = tf.where(p_brightness > 0.5,
                 tf.clip_by_value(x + delta, -0.5, 0.5),
                 x)
    
    p_contrast = tf.random.uniform(rn_shape, 0.0, 1.0)
    alpha = tf.random.uniform(rn_shape, 0.8, 1.2)
    x = tf.where(p_contrast > 0.5,
                 tf.clip_by_value(x * alpha, -0.5, 0.5),
                 x)
        
    p_rot = tf.random.uniform((), 0.0, 1.0)
    if p_rot > 0.5:
        x = rotate(x)

    p_shift = tf.random.uniform((), 0.0, 1.0)
    if p_shift > 0.5:
        x = shift(x)

    x = tf.clip_by_value(x, -0.5, 0.5)
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
    parser.add_argument('--output-path', type=str, default='cifar100/base-relu-vgg16', required=False)
    parser.add_argument('--seed', type=int, default=3, required=False)
    parser.add_argument('--batch-size', type=int, default=32, required=False)
    parser.add_argument('--learning-rate', type=float, default=0.0001, required=False)
    parser.add_argument('--epochs', type=int, default=300, required=False)
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

    input_shape = INPUT_SHAPE + [3]
    n_classes = 100

    model = vgg16(n_classes, input_shape, dropout=True)
    model.summary()

    lr = args.learning_rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    ds_train, ds_test = tfds.load('cifar100',
                                  split=['train', 'test'],
                                  shuffle_files=True,
                                  as_supervised=True,
                                  with_info=False,
                                  download=False) # Change this if necessary

    ds_train = ds_train.cache().shuffle(10000).batch(batch_size, drop_remainder=True).map(preprocess_input).map(lambda x, y: augment(x, y, batch_size)).repeat().prefetch(tf.data.AUTOTUNE)
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

# Test acc: 71.17%