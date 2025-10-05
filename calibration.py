from copy import copy

import tensorflow as tf
import numpy as np

from sklearn.cluster import KMeans
from functools import partial

from typing import Callable, List, Optional, Sequence, Tuple
from layers import QuantizedReLU, QuantizedConv2D, QuantizedDense, QuantizedInput


def fold_batch_norm_layers(model: tf.keras.Model, model_creation_func: Callable) -> tf.keras.Model:
    """
    Fuses batch norm parameters with weights and biases of the preceding conv or dense layer.

    NOTE: For models architectures not used in this repository this method may require modifications.
    """
    inp_shape = model.input.shape
    out_shape = model.outputs[0].shape
    
    weighted_layers_names = [l.name for l in model.layers if 'conv' in l.name or 'dense' in l.name]
    bnorm_layers = [l for l in model.layers if 'norm' in l.name]

    # Skip output matmul layer
    new_trainable_params = [None] * (len(weighted_layers_names) - 1)
    for bn_layer in bnorm_layers:
        mean = bn_layer.moving_mean
        var = bn_layer.moving_variance
        gamma, beta = bn_layer.trainable_weights
        eps = bn_layer.epsilon

        w_layer = bn_layer._inbound_nodes[0].inbound_layers

        w, b = w_layer.trainable_weights

        gamma_over_var = gamma / tf.math.sqrt(var + eps)
        new_w = gamma_over_var * w
        new_b = gamma_over_var * b - gamma_over_var * mean + beta

        idx = weighted_layers_names.index(w_layer.name)
        new_trainable_params[idx] = [new_w, new_b]

    assert all(t is not None for t in new_trainable_params), [type(t) for t in new_trainable_params]

    # Flatten list of lists
    new_tp = []
    for p in new_trainable_params:
        new_tp.extend(p)
    new_trainable_params = new_tp
    
    new_trainable_params.extend(model.get_weights()[-2:])

    fused_bn_model = model_creation_func(out_shape[-1], inp_shape[1:], batch_norm=False)
    fused_bn_model.set_weights(new_trainable_params)

    return fused_bn_model


def add_relu_outputs_to_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Add ReLU outputs as additional outputs to keras model.
    """
    relu_layers = [l for l in model.layers if 're_lu' in l.name]
    relu_outputs = [l.output for l in relu_layers]
    return tf.keras.Model(inputs=model.input, outputs=[model.output] + relu_outputs), relu_layers


def get_ranges_by_clustering(min_vals: List[float], max_vals: List[float], max_k: int,
                             inertia_decrease_thresh: float, seed: int, outlier_idxs: Optional[int] = None):
    """
    This function performs clustering on min-max value pairs and returns lists of min and max values
    of their corresponding clusters, as well as mapping of elements to clusters (list of equal length
    to number of min/max vals containing cluster indices starting from zero and ending on chosen k - 1).
    """
    assert len(min_vals) == len(max_vals)
    assert len(min_vals) >= max_k
    min_max_vals = list(zip(min_vals, max_vals))
    min_max_vals = [np.array(vals) for vals in min_max_vals]
    if outlier_idxs is not None:
        old_min_max_vals = copy(min_max_vals)
        new_min_max_vals = [v for c, v in enumerate(min_max_vals) if c not in outlier_idxs]
        min_max_vals = new_min_max_vals

    # Calculate kmeans for different cluster numbers   
    kmeans_post_fit = []
    inertias = []
    ks = []
    for k in range(1, max_k):
        fit_kmeans = KMeans(n_clusters=k, random_state=seed).fit(min_max_vals)
        kmeans_post_fit.append(fit_kmeans)
        inertias.append(fit_kmeans.inertia_)
        ks.append(k)

    # Get best clusters number index
    max_inertia = inertias[0]
    prev_inertia = max_inertia
    chosen_inertia_idx = 0
    for c, inertia in enumerate(inertias[1:]):
        inertia_decrease_perc = (prev_inertia - inertia) / max_inertia
        if inertia_decrease_perc > inertia_decrease_thresh:
            chosen_inertia_idx = c + 1
            prev_inertia = inertia
        else:
            break

    if chosen_inertia_idx == 0:
        raise RuntimeError('Best cluster configuration choice failed')

    # Calculate min-max values for each
    # cluster and form new min and max lists
    chosen_kmeans_setup = kmeans_post_fit[chosen_inertia_idx]
    clusters_mapping = chosen_kmeans_setup.labels_
    chosen_k = ks[chosen_inertia_idx]
    cluster_indices = list(range(0, chosen_k))
    num_pairs = len(min_max_vals)
    range_arr = np.arange(0, num_pairs, 1, dtype=np.int32)
    
    new_min_vals = np.array([0.0] * num_pairs, dtype=np.float32)
    new_max_vals = np.array([0.0] * num_pairs, dtype=np.float32)
    for c_idx in cluster_indices:
        label_map = c_idx == clusters_mapping
        c_mem_indices = range_arr[label_map].tolist()
        c_mins = [min_max_vals[idx][0] for idx in c_mem_indices]
        c_maxs = [min_max_vals[idx][1] for idx in c_mem_indices]
        c_min = min(c_mins)
        c_max = max(c_maxs)
        new_min_vals[label_map] = c_min
        new_max_vals[label_map] = c_max

    new_min_vals = new_min_vals.tolist()
    new_max_vals = new_max_vals.tolist()
    if outlier_idxs is not None:
        num_vals = len(old_min_max_vals)
        rng = np.arange(0, num_vals, 1)
        max_cluster_idx = max(clusters_mapping)
        outliers_no = len(outlier_idxs)
        separate_clusters_idxs = list(range(max_cluster_idx + 1, max_cluster_idx + 1 + outliers_no, 1))
        mi_v = [None] * num_vals
        ma_v = [None] * num_vals
        cl_m = [None] * num_vals
        idx = 0
        for i in rng.tolist():
            if i not in outlier_idxs:
                mi_v[i] = new_min_vals[idx]
                ma_v[i] = new_max_vals[idx]
                cl_m[i] = clusters_mapping[idx]
                idx += 1
            else:
                mi_v[i] = old_min_max_vals[i][0]
                ma_v [i] = old_min_max_vals[i][1]
                cl_m[i] = separate_clusters_idxs[0]
                del separate_clusters_idxs[0]
        
        new_min_vals = mi_v
        new_max_vals = ma_v
        clusters_mapping = cl_m
    
    return new_min_vals, new_max_vals, clusters_mapping


def find_nearest_matmul_layer_name(layer):
    name = layer.name
    if 'conv' in name or \
       'dense' in name:
        return [name]
    elif 'add' in name:
        return []
    else:
        out_nodes = layer._outbound_nodes
        if len(out_nodes) == 0:
            raise ValueError('Nearest matmul layer for ReLU not found.')
        matmul_names = []
        for node in out_nodes:
            out_layer = node.outbound_layer
            name =  find_nearest_matmul_layer_name(out_layer)
            if not isinstance(name, list):
                name = [name]

            matmul_names.extend(name)

        return matmul_names


def calibrate(model: tf.keras.Model, calibration_ds: tf.data.Dataset,
              num_quantization_levels: int, model_creation_func: Callable,
              input_range: Tuple[float, float] = (-0.5, 0.5),
              mode: str = 'cluster', max_clusters: int | Sequence[int] = 8,
              inertia_decrease_perc: float = 0.15, a_outlier_idxs: Optional[int] = None,
              w_outlier_idxs: Optional[int] = None, random_seed: int = 1234) -> tf.keras.Model:
    """
    This method calculates min and max values for each ReLU layer in the model based on calibration_ds,
    calculates min and max values for weights and biases in each layer and produces a model that performs
    fake quantization in the float domain in each dense, conv and relu layer to allow quantization-aware training.
    For matmul layers only weights are quantized. Additionally min and max values for input layers must be provided
    as this type of partial quantization also adds additional quantization of input to the network.
    
    Notable args:
    - mode: either 'uniform' or 'cluster'.
            a. When set to 'uniform' all activations and weights are quantized on separate global min-max scales.
            b. When set to 'cluster' all activations and weights are quantized on min-max scales taken from k-means
               clusterings' clusters chosen by the automated elbow method. k-means clustering is performed on weights' and
               activations' min-max values calculated from the calibration dataset. Clusters initialization is performed
               using k-means++ method.
            NOTE: 'uniform' mode requires more quantization levels to work well and thus increases memory needs.
    - max_clusters: effective only when mode is set to 'cluster', sets maximum number of clusters to analyze using the elbow method.
                    It can either be int or a sequence of two ints for activations and weights separately.
    - inertia_decrease_perc: effective only when mode is set to 'cluster', cluster numbers with % decrease of inertia below that values
                             will be rejected. Last cluster number with inertia decrease above that threshold is chosen. Percentage is
                             calculated based on inertia from k = 1.
    - a_outlier_idxs and w_outlier_idxs: indices of activations and weights layers to not be included in the clustering process and
                                         be put into separate clusters.

    NOTE: For models architectures not used in this repository this method may require modifications.
    NOTE: List of supported layers that change resolution of tensors in models:
          - Max pooling 2D
          - Global Average Pooling 2D
          - Upsampling 2D with nearest neighbours interpolation
          - Flatten
    """
    assert mode == 'uniform' or mode == 'cluster'

    if mode == 'cluster' and not isinstance(max_clusters, Sequence):
        max_clusters = [max_clusters, max_clusters]

    model, relus = add_relu_outputs_to_model(model)

    # Get activations min-max values
    num_relus = len(model.outputs) - 1
    min_activation_vals = [None] * num_relus
    max_activation_vals = [None] * num_relus
    for example in calibration_ds:
        inp = example[0]
        relu_outputs = model(inp, training=False)[1:]

        for c, relu_out in enumerate(relu_outputs):
            curr_min = tf.reduce_min(relu_out).numpy()
            curr_max = tf.reduce_max(relu_out).numpy()

            if  min_activation_vals[c] is None or min_activation_vals[c] > curr_min:
                min_activation_vals[c] = curr_min

            if max_activation_vals[c] is None or max_activation_vals[c] < curr_max:
                max_activation_vals[c] = curr_max
    
    if mode == 'uniform':
        min_val = min(min_activation_vals)
        max_val = max(max_activation_vals)
        n_activations = len(min_activation_vals)
        min_activation_vals = [min_val] * n_activations
        max_activation_vals = [max_val] * n_activations
        a_clusters_map = [0] * n_activations
    else:
        min_activation_vals, max_activation_vals, a_clusters_map = get_ranges_by_clustering(
            min_activation_vals, max_activation_vals, max_clusters[0],
            inertia_decrease_perc, random_seed, a_outlier_idxs)
        
    quantized_activations = []
    for min_val, max_val, cluster_idx in zip(min_activation_vals, max_activation_vals, a_clusters_map):
        activation = QuantizedReLU(num_quantization_levels, min_val, max_val, cluster_idx)
        quantized_activations.append(activation)

    # Get weights min-max values
    conv_layers = [l for l in model.layers if 'conv' in l.name]
    dense_layers = [l for l in model.layers if 'dense' in l.name]
    matmul_layers = conv_layers + dense_layers
    matmul_names = [m.name for m in matmul_layers]
    
    w_min_per_layer = []
    w_max_per_layer = []
    for mm_layer in matmul_layers:
        w_min_per_layer.append(tf.reduce_min(mm_layer.kernel).numpy())
        w_max_per_layer.append(tf.reduce_max(mm_layer.kernel).numpy())
    
    if mode == 'uniform':
        min_val = min(w_min_per_layer)
        max_val = max(w_max_per_layer)
        n_weights = len(w_min_per_layer)
        w_min_per_layer = [min_val] * n_weights
        w_max_per_layer = [max_val] * n_weights
        w_clusters_map = [0] * n_weights
    else:
        w_min_per_layer, w_max_per_layer, w_clusters_map = get_ranges_by_clustering(
            w_min_per_layer, w_max_per_layer, max_clusters[1],
            inertia_decrease_perc, random_seed, w_outlier_idxs)

    def quant_wrapper(quant_layer, num_quantization_levels, w_min, w_max, *args, **kwargs):
        return quant_layer(num_quantization_levels, w_min, w_max, *args, **kwargs)
    
    num_matmul_layers = len(matmul_layers)
    num_conv_layers = len(conv_layers)
    quantized_matmul_layers = []
    for i in range(num_matmul_layers):
        quant_layer = QuantizedConv2D if i < num_conv_layers else QuantizedDense
        quantized_matmul_layers.append(partial(quant_wrapper,
                                               quant_layer,
                                               num_quantization_levels,
                                               w_min_per_layer[i],
                                               w_max_per_layer[i],
                                               w_clusters_map[i]))

    inp_shape = model.input.shape
    out_shape = model.outputs[0].shape
    input_quantization_layer = QuantizedInput(num_quantization_levels, input_range[0], input_range[1])
    quantized_model = model_creation_func(out_shape[-1],
                                          inp_shape[1:],
                                          quantized_activations,
                                          quantized_matmul_layers,
                                          batch_norm=False,
                                          input_processing_layer=input_quantization_layer)
    quantized_model.set_weights(model.get_weights())
    
    relu_matmul_pairs = []
    for c, r in enumerate(relus):
        out_layers = [node.outbound_layer for node in r._outbound_nodes]
        for out_l in out_layers:
            out_names = find_nearest_matmul_layer_name(out_l)
            if len(out_names) == 0:
                continue

            for out_n in out_names:
                idx = matmul_names.index(out_n)
                relu_matmul_pairs.append((a_clusters_map[c], w_clusters_map[idx]))

    num_unique_cluster_pairs = len(set(relu_matmul_pairs))
    additional_pairs = 1 if not 'unet' in model.name else 2
    print('\nNumber of activations-weights unique groups:', additional_pairs + num_unique_cluster_pairs)
    print('This number influences the number of parameters cached into memory.\n')

    return quantized_model
