import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path
import argparse
import shutil
import sys
import os

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.absolute()))

from models import resnet32
from calibration import fold_batch_norm_layers, calibrate

INPUT_SHAPE = [224, 224]


@tf.function
def augment(x, y, bs):
    """
    Augmentations in order:
    1. Flip left-right
    2. Flip up-down
    3. Brightness
    4. Contrast
    """
    rn_shape = (bs, 1, 1, 1)

    p_flip = tf.random.uniform(rn_shape, 0.0, 1.0)
    x = tf.where(p_flip > 0.5,
                 tf.image.flip_left_right(x),
                 x)
    
    p_flip = tf.random.uniform(rn_shape, 0.0, 1.0)
    x = tf.where(p_flip > 0.5,
                 tf.image.flip_up_down(x),
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
        
    x = tf.clip_by_value(x, -0.5, 0.5)
    return x, y


def preprocess_input(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x -= 0.5
    x = tf.image.resize(x, INPUT_SHAPE, 'bicubic')
    y = tf.cast(y, tf.float32)
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
    parser.add_argument('--output-path', type=str, default='malaria/quant-relu-resnet32', required=False)
    parser.add_argument('--input-path', type=str, default='malaria/base-relu-resnet32', required=False)
    parser.add_argument('--bits', type=int, default=8, required=False)
    parser.add_argument('--seed', type=int, default=3, required=False)
    parser.add_argument('--batch-size', type=int, default=32, required=False)
    parser.add_argument('--learning-rate', type=float, default=0.0001, required=False)
    parser.add_argument('--epochs', type=int, default=6, required=False)
    parser.add_argument('--steps-per-epoch', type=int, default=300, required=False)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    input_path = Path(args.input_path)
    bits = args.bits

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path)
    
    seed = args.seed
    tf.keras.utils.set_random_seed(seed)

    batch_size = args.batch_size
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    validation_steps = None

    ds_train, ds_test = tfds.load('malaria',
                                  split=['train[:80%]', 'train[80%:]'],
                                  shuffle_files=True,
                                  as_supervised=True,
                                  with_info=False,
                                  download=False) # Change this if necessary

    ds_train = ds_train.cache().shuffle(10000).map(preprocess_input).batch(batch_size, drop_remainder=True).map(lambda x, y: augment(x, y, batch_size)).repeat().prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_input).batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)

    model = resnet32(1, INPUT_SHAPE + [3])
    model.load_weights(input_path / 'model' / 'weights.h5')

    folded_bn_model = fold_batch_norm_layers(model, resnet32)
    quantized_model = calibrate(folded_bn_model, ds_test, 2**bits, resnet32, (-0.5, 0.5), 'cluster', 8, inertia_decrease_perc=0.30, random_seed=seed)
    quantized_model.summary()

    quantized_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                            loss=tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size'),
                            metrics=[tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.BinaryAccuracy()])

    ptq_eval = quantized_model.evaluate(ds_test)
    print('\nPost-quantization ROC AUC:', ptq_eval[1])
    print('\nPost-quantization accuracy:', ptq_eval[2])

    checkpoints_path = str(output_path / 'model')
    os.makedirs(checkpoints_path)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path, monitor='val_auc',
                                                              save_weights_only=False, save_best_only=True, mode='max')
    
    log_dir = output_path / 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [checkpoints_callback, tensorboard_callback]
    history = quantized_model.fit(x=ds_train, validation_data=ds_test, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps, callbacks=callbacks)

    history_dict = history.history
    loss_list = history_dict['val_loss']
    auc_list = history_dict['val_auc']
    acc_list = history_dict['val_binary_accuracy']
    best_auc_idx = auc_list.index(max(auc_list))
    
    tf.keras.models.save_model(quantized_model, output_path / 'keras_model.h5', save_format='h5')

    print('\n')
    print('Test loss:', loss_list[best_auc_idx])
    print('Test ROC AUC:', auc_list[best_auc_idx])
    print('Test accuracy:', acc_list[best_auc_idx])

# Test roc auc: 99.30%
# Test acc: 96.17% 
# Number of clusters: 4
# Cached array size: 1 MB
