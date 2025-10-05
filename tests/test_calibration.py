import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from calibration import *
from models import lenet5
from mnist.train_lenet5 import preprocess_input

tf.keras.utils.set_random_seed(3456)

model = tf.keras.models.load_model('tests/base-relu-lenet5/keras_model.h5')


def test_batch_norm_folding():
    model.summary()

    folded_bn_model = fold_batch_norm_layers(model, lenet5)
    folded_bn_model.summary()

    inp = tf.random.uniform([4, 28, 28, 1], -0.5, 0.5, tf.float32)
    mod_out = model(inp).numpy()
    folded_bn_mod_out = folded_bn_model(inp).numpy()

    np.testing.assert_almost_equal(mod_out, folded_bn_mod_out, 3)


def test_get_ranges_by_clustering():
    min_vals = [-0.05, -0.15, -0.10, -0.15, -0.03]
    max_vals = [0.05, 0.15, 0.15, 0.10, 0.03]
    max_k = 5
    inertia_decrease_thresh = 0.15
    seed = 1234

    exp_min_vals = np.array([-0.05, -0.15, -0.15, -0.15, -0.05])
    exp_max_vals = np.array([0.05, 0.15, 0.15, 0.15, 0.05])
    exp_mapping = np.array([0, 1, 1, 1, 0])

    new_min_vals, new_max_vals, mapping = get_ranges_by_clustering(min_vals, max_vals, max_k, inertia_decrease_thresh, seed)

    new_min_vals = np.array(new_min_vals)
    new_max_vals = np.array(new_max_vals)
    mapping = np.array(mapping)

    np.testing.assert_almost_equal(new_min_vals, exp_min_vals, 2)
    np.testing.assert_almost_equal(new_max_vals, exp_max_vals, 2)
    np.testing.assert_equal(mapping, exp_mapping)


def test_calibrate():
    model.summary()

    folded_bn_model = fold_batch_norm_layers(model, lenet5)
    folded_bn_model.summary()

    _, ds_test = tfds.load('mnist',
                           split=['train', 'test'],
                           shuffle_files=True,
                           as_supervised=True,
                           with_info=False,
                           download=False) # Change this if necessary

    ds_test = ds_test.batch(500).map(preprocess_input).cache().prefetch(tf.data.AUTOTUNE)

    num_quant_levels = 256
    max_clusters = (4, 5)
    inertia_perc_thresh = 0.15
    seed = 1234
    for mode in ['uniform', 'cluster']:
        print(mode)

        calibrated_model = calibrate(folded_bn_model, ds_test, num_quant_levels, lenet5,
                                     (-0.5, 0.5), mode, max_clusters, inertia_perc_thresh,
                                     None, None, seed)
        calibrated_model.summary()

        layers_prefixes_in_sequence = ['input', 'quantized_input', 'quantized_conv2d', 'quantized_re_lu', 'max_pooling2d',
                                       'quantized_conv2d', 'quantized_re_lu', 'max_pooling2d',
                                       'flatten', 'quantized_dense', 'quantized_re_lu',
                                       'quantized_dense', 'quantized_re_lu', 'quantized_dense']
        for c, layer in enumerate(calibrated_model.layers):
            assert layers_prefixes_in_sequence[c] in layer.name

        if mode == 'uniform':
            matmul_layers = [l for l in folded_bn_model.layers if 'conv' in l.name or 'dense' in l.name]
            min_w_val = min([tf.reduce_min(l.kernel).numpy() for l in matmul_layers])
            max_w_val = max([tf.reduce_max(l.kernel).numpy() for l in matmul_layers])
        
            c_matmul_layers = [l for l in calibrated_model.layers if 'conv' in l.name or 'dense' in l.name]
            for c_matmul in c_matmul_layers:
                np.testing.assert_almost_equal(min_w_val, c_matmul.min_w_value, 2)
                np.testing.assert_almost_equal(max_w_val, c_matmul.max_w_value, 2)

            min_val = 0.0
            max_val = 7.382706
            c_relu_layers = [l for l in calibrated_model.layers if 're_lu' in l.name]
            for c_relu in c_relu_layers:
                np.testing.assert_almost_equal(min_val, c_relu.min_value, 2)
                np.testing.assert_almost_equal(max_val, c_relu.max_value, 2)
        else:
            min_w_vals = [-1.55874502658844, -0.39100372791290283, -0.39100372791290283, -0.7953603267669678, -0.7953603267669678]
            max_w_vals = [1.1724786758422852, 0.2997717261314392, 0.2997717261314392, 0.7226521968841553, 0.7226521968841553]
            w_mapping = [0, 1, 1, 2, 2]
            c_matmul_layers = [l for l in calibrated_model.layers if 'conv' in l.name or 'dense' in l.name]
            for min_w_val, max_w_val, c_idx, c_matmul in zip(min_w_vals, max_w_vals, w_mapping, c_matmul_layers):
                np.testing.assert_almost_equal(min_w_val, c_matmul.min_w_value, 2)
                np.testing.assert_almost_equal(max_w_val, c_matmul.max_w_value, 2)
                assert c_idx == c_matmul.cluster_idx
            
            min_a_vals = [0.0, 0.0, 0.0, 0.0]
            max_a_vals = [5.70822811126709, 5.70822811126709, 5.70822811126709, 7.382706165313721]
            a_mapping = [0, 0, 0, 1]
            c_relu_layers = [l for l in calibrated_model.layers if 're_lu' in l.name]
            for min_val, max_val, c_idx, c_relu in zip(min_a_vals, max_a_vals, a_mapping, c_relu_layers):
                np.testing.assert_almost_equal(min_val, c_relu.min_value, 2)
                np.testing.assert_almost_equal(max_val, c_relu.max_value, 2)
                assert c_idx == c_relu.cluster_idx
