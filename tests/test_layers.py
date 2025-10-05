import tensorflow as tf
import numpy as np

from layers import *

tf.keras.utils.set_random_seed(3456)


def test_perform_binary_search_quantized_input():
    """Simulation of binary search spiking neuron forward pass for QuantizedInput layer."""
    bits = 8
    min_activ_value = -5.0
    max_activ_value = 5.0

    num_quant_levels = 2**bits
    q_step = (max_activ_value - min_activ_value) / (num_quant_levels - 1)
    
    idx_div = num_quant_levels
    index_offsets = []
    for _ in range(bits):
        idx_div /= 2
        index_offsets.append(int(idx_div))
    
    interval_halves = []
    for b in range(bits, 0, -1):
        interval_halves.append((2**b - 1) * q_step / 2.0)

    v_init_offsets = []
    for int_h in interval_halves:
        v_init_offsets.append(int_h + interval_halves[-1])

    # Include negative values to test if they get assigned an index 0
    # Include values above max to see if they get assigned an index 255
    input_tensor = tf.random.uniform([4, 20], -6.0, 6.0)
    idx_offsets = perform_binary_search(input_tensor, bits, min_activ_value,
                                        interval_halves, v_init_offsets, index_offsets)
    idx_offsets = idx_offsets.numpy()

    input_tensor = tf.where(input_tensor < min_activ_value, min_activ_value, input_tensor)
    input_tensor = tf.where(input_tensor > max_activ_value, max_activ_value, input_tensor)
    exp_idx_offsets = tf.cast(tf.round((input_tensor - min_activ_value) / q_step), tf.int32).numpy()
    np.testing.assert_equal(idx_offsets, exp_idx_offsets)


def test_binary_search_snn_dense_layer_w_relu():
    bits = 3
    min_activ_value = 0.0
    max_activ_value = 5.0
    
    input_tensor = tf.random.uniform([8, 4], -1.0, 6.0) # N x Win shape
    num_q_levels = 2**bits
    cached_products = tf.linspace(-0.5, 0.5, num_q_levels)[tf.newaxis, :] *\
                      tf.linspace(min_activ_value, max_activ_value, num_q_levels + 1)[1:, tf.newaxis] # 0 is skipped

    w_map = np.array([[0, 0, 1], [1, 2, 3], [3, 7, 7], [2, 5, 5]], dtype=np.int32) # Win x Wout shape

    # NOTE: No bias to check correct values
    num_q_levels = 2**bits
    bs_snn_layer = BinarySearchSNNDense(w_map, min_activ_value, max_activ_value, bits, is_relu=True)
    res = bs_snn_layer(input_tensor, cached_products)
    
    exp_res_b_zero = np.array([[-0.9375,     -1.1160715,  -0.3571427, -0.26785713],
                               [-0.9375,     -0.6696428,  2.5,        0.26785716],
                               [-0.66964287, -0.22321418, 2.5,        0.26785716]]).sum(axis=-1)
    exp_res_b_five = np.array([[0.0, -0.2232143, -0.31249985, 0.0],
                               [0.0, -0.13392857, 2.1875, 0.0],
                               [0.0, -0.04464284, 2.1875, 0.0]]).sum(axis=-1)

    np.testing.assert_almost_equal(res[0, ...], exp_res_b_zero, 4)
    np.testing.assert_almost_equal(res[5, ...], exp_res_b_five, 4)


def test_perform_binary_search_relu():
    """Simulation of binary search spiking neuron forward pass for ReLU layer"""
    bits = 8
    min_activ_value = 0.0
    max_activ_value = 5.0

    num_quant_levels = 2**bits
    q_step = (max_activ_value - min_activ_value) / num_quant_levels

    idx_div = num_quant_levels
    index_offsets = []
    for _ in range(bits):
        idx_div /= 2
        index_offsets.append(int(idx_div))
    
    interval_halves = []
    for b in range(bits, 0, -1):
        interval_halves.append((2**b - 1) * q_step / 2.0)

    v_init_offsets = []
    for int_h in interval_halves:
        v_init_offsets.append(int_h + interval_halves[-1])


    # Include negative values to test if values below q_step will get an index of 0
    # Include values above max to see if they get assigned an index 255
    input_tensor = tf.random.uniform([4, 20], -1.0, 6.0)

    # We have to shift min val because we exclude 0 from the cached table
    min_val = min_activ_value + q_step
    idx_offsets = perform_binary_search(input_tensor, bits, min_val,
                                        interval_halves, v_init_offsets, index_offsets)
    idx_offsets = idx_offsets.numpy()
    input_tensor = tf.where(input_tensor < min_activ_value, min_activ_value, input_tensor)
    input_tensor = tf.where(input_tensor > max_activ_value, max_activ_value, input_tensor)

    # Expected output indices are shifted by 1 because we skipped 0 in the cached table
    exp_idx_offsets = tf.cast(tf.clip_by_value(tf.round((input_tensor - min_activ_value) / q_step) - 1,
                                               0, num_quant_levels - 1),
                              tf.int32).numpy()
    np.testing.assert_equal(idx_offsets, exp_idx_offsets)


def test_binary_search_snn_dense_layer_w_quantized_input():
    bits = 3
    min_activ_value = -0.5
    max_activ_value = 0.5
    
    input_tensor = tf.random.uniform([8, 4], -0.7, 0.7) # N x Win shape
    num_q_levels = 2**bits
    cached_products = tf.linspace(-0.5, 0.5, num_q_levels)[tf.newaxis, :] *\
                      tf.linspace(min_activ_value, max_activ_value, num_q_levels)[:, tf.newaxis]

    w_map = np.array([[0, 0, 1], [1, 2, 3], [3, 7, 7], [2, 5, 5]], dtype=np.int32) # Win x Wout shape

    # NOTE: No bias to check correct values
    num_q_levels = 2**bits
    bs_snn_layer = BinarySearchSNNDense(w_map, min_activ_value, max_activ_value, bits, is_relu=False)
    res = bs_snn_layer(input_tensor, cached_products)
    
    exp_res_b_zero = np.array([[-0.107142866,  -0.127551049,  -0.0255101975, 0.107142851],
                               [-0.107142866,  -0.0765306205, 0.178571463,   -0.107142866],
                               [-0.0765306205, -0.0255101975, 0.178571463,   -0.107142866]]).sum(axis=-1)
    exp_res_b_five = np.array([[0.25,        0.076530613,  -0.0357142687, 0.107142851],
                               [0.25,        0.0459183604, 0.25,          -0.107142866],
                               [0.178571433, 0.0153061142, 0.25,          -0.107142866]]).sum(axis=-1)

    np.testing.assert_almost_equal(res[0, ...], exp_res_b_zero, 4)
    np.testing.assert_almost_equal(res[5, ...], exp_res_b_five, 4)


q_model = tf.keras.models.load_model('tests/quant-relu-fc2/keras_model.h5', custom_objects={'QuantizedInput': QuantizedInput,
                                                                                            'QuantizedDense': QuantizedDense,
                                                                                            'QuantizedReLU': QuantizedReLU})


def test_binary_search_snn_equivalency_to_quantized_dense_w_quantized_input():
    bits = 8
    input_layer = q_model.get_layer('quantized_input')
    dense_layer_one = q_model.get_layer('quantized_dense')
    
    num_q_levels = 2**bits

    a_two = tf.linspace(input_layer.min_value, input_layer.max_value, num_q_levels)[:, tf.newaxis] 
    w_two = tf.linspace(dense_layer_one.min_w_value, dense_layer_one.max_w_value, num_q_levels)[tf.newaxis, :]
    cached_prod_two = a_two * w_two
    
    q_w_two = tf.clip_by_value(dense_layer_one.kernel, dense_layer_one.min_w_value, dense_layer_one.max_w_value) - dense_layer_one.min_w_value
    w_map_two = tf.cast(tf.math.round(q_w_two / dense_layer_one.w_quantization_step), tf.int32).numpy()
    snn_layer_two = BinarySearchSNNDense(w_map_two, input_layer.min_value, input_layer.max_value, bits, dense_layer_one.bias, is_relu=False)
     
    inp = tf.random.uniform([4, dense_layer_one.kernel.shape[0]], -0.6, 0.6, dtype=tf.float32)
    snn_out = snn_layer_two(inp, cached_prod_two)
    ann_out = input_layer(inp)
    ann_out = dense_layer_one(ann_out)

    np.testing.assert_almost_equal(snn_out.numpy(), ann_out.numpy(), 3)


def test_binary_search_snn_equivalency_to_quantized_dense_w_quantized_relu():
    bits = 8
    relu_layer = q_model.get_layer('quantized_re_lu')
    dense_layer_two = q_model.get_layer('quantized_dense_1')
    
    num_q_levels = 2**bits
    
    # We skip zero here since it doesn't point to any values cached in memory
    a_two = tf.linspace(relu_layer.min_value, relu_layer.max_value, num_q_levels + 1)[1:, tf.newaxis]
    w_two = tf.linspace(dense_layer_two.min_w_value, dense_layer_two.max_w_value, num_q_levels)[tf.newaxis, :]
    cached_prod_two = a_two * w_two
    
    q_w_two = tf.clip_by_value(dense_layer_two.kernel, dense_layer_two.min_w_value, dense_layer_two.max_w_value) - dense_layer_two.min_w_value
    w_map_two = tf.cast(tf.math.round(q_w_two / dense_layer_two.w_quantization_step), tf.int32).numpy()
    snn_layer_two = BinarySearchSNNDense(w_map_two, relu_layer.min_value, relu_layer.max_value, bits, dense_layer_two.bias, is_relu=True)

    inp = tf.random.uniform([4, dense_layer_two.kernel.shape[0]], -0.6, 0.6, dtype=tf.float32)
    snn_out = snn_layer_two(inp, cached_prod_two)
    snn_out = dense_layer_two.activation(snn_out) # Add final activation since it's the last layer
    ann_out = relu_layer(inp)
    ann_out = dense_layer_two(ann_out)

    np.testing.assert_almost_equal(snn_out.numpy(), ann_out.numpy(), 3)
