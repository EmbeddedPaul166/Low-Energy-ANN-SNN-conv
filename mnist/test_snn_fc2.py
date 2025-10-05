import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path
import argparse
import shutil
import sys
import os

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.absolute()))

from layers import BinarySearchSNNDense, QuantizedInput, QuantizedDense, QuantizedReLU


def convert_to_snn(model, bits):
    inp_layer = model.get_layer('quantized_input')
    dense_layer_one = model.get_layer('quantized_dense')
    relu_layer = model.get_layer('quantized_re_lu')
    dense_layer_two = model.get_layer('quantized_dense_1')

    # Calculate cached products of activations and weights

    num_q_levels = 2**bits
    a_one = tf.linspace(inp_layer.min_value, inp_layer.max_value, num_q_levels)[:, tf.newaxis]
    w_one = tf.linspace(dense_layer_one.min_w_value, dense_layer_one.max_w_value, num_q_levels)[tf.newaxis, :]
    cached_prod_one = a_one * w_one

    # We skip zero here since it doesn't point to any values cached in memory
    a_two = tf.linspace(relu_layer.min_value, relu_layer.max_value, num_q_levels + 1)[1:, tf.newaxis]
    w_two = tf.linspace(dense_layer_two.min_w_value, dense_layer_two.max_w_value, num_q_levels)[tf.newaxis, :]
    cached_prod_two = a_two * w_two
    
    # Calculate maps indicating partial indices of weights in cached products
    q_w_one = tf.clip_by_value(dense_layer_one.kernel, dense_layer_one.min_w_value, dense_layer_one.max_w_value) - dense_layer_one.min_w_value
    w_map_one = tf.cast(tf.math.round(q_w_one / dense_layer_one.w_quantization_step), tf.int32).numpy()
    
    q_w_two = tf.clip_by_value(dense_layer_two.kernel, dense_layer_two.min_w_value, dense_layer_two.max_w_value) - dense_layer_two.min_w_value
    w_map_two = tf.cast(tf.math.round(q_w_two / dense_layer_two.w_quantization_step), tf.int32).numpy()

    # Construct SNN model
    snn_layer_one = BinarySearchSNNDense(w_map_one, inp_layer.min_value, inp_layer.max_value, bits, dense_layer_one.bias, is_relu=False)
    snn_layer_two = BinarySearchSNNDense(w_map_two, relu_layer.min_value, relu_layer.max_value, bits, dense_layer_two.bias, is_relu=True)

    inp = model.input
    x = snn_layer_one(inp, cached_prod_one)
    x = snn_layer_two(x, cached_prod_two)
    out = dense_layer_two.activation(x)

    return tf.keras.Model(inputs=inp, outputs=out, name='fc2_snn_model')


def preprocess_input(x, y):
    x = tf.reshape(x, [tf.shape(x)[0], -1])
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
    parser.add_argument('--output-path', type=str, default='mnist/snn-relu-fc2', required=False)
    parser.add_argument('--input-path', type=str, default='mnist/quant-relu-fc2', required=False)
    parser.add_argument('--bits', type=int, default=8, required=False)
    parser.add_argument('--seed', type=int, default=3, required=False)
    parser.add_argument('--batch-size', type=int, default=500, required=False)
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
    validation_steps = None

    _, ds_test = tfds.load('mnist',
                           split=['train', 'test'],
                           shuffle_files=True,
                           as_supervised=True,
                           with_info=False,
                           download=False) # Change this if necessary

    ds_test = ds_test.batch(batch_size).map(preprocess_input).cache().prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(input_path / 'keras_model.h5', custom_objects={'QuantizedInput': QuantizedInput,
                                                                                      'QuantizedDense': QuantizedDense,
                                                                                      'QuantizedReLU': QuantizedReLU})
    model.summary()

    snn_model = convert_to_snn(model, bits)
    snn_model.summary()

    snn_model.compile(metrics=[tf.keras.metrics.CategoricalAccuracy()])

    print('\nPost-conversion accuracy:', snn_model.evaluate(ds_test)[1])

# Test acc: 98.38%
#
# Original quantized network achieved 98.42%. Small drop of performance
# may come from different operation kernels being used by tensorflow
# and the fact that all of the result were obtained on RTX 3080 GPU
# (CPUs usually have better result reproducibility).