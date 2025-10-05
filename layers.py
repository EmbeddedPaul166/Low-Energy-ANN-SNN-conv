from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf

from keras.src.utils import conv_utils


def quant_dequant_float(x, min_value: float, max_value: float, quantization_step: float, is_relu: bool = False): 
    """
    Performs quantize-dequantize operations in the floating point domain.
    Quantization in floating point is equivalent to rounding float values to
    the closest quantization level where levels are uniformly distributed across an interval:
    <min_value, max_value>.

    This fake-quantization block also performs an additional action to keep results identical
    to the SNN post-conversion: for ReLU layers it additionally sets all values below
    min_value + quantization_step to zero. This is performed because cached products tables
    don't contain zeros in them to avoid redundancy and therefore all values below  the minimum
    non-zero value are set to zero.
    """
    x = tf.clip_by_value(x, min_value, max_value) - min_value
    if is_relu:
        x = tf.where(x < min_value + quantization_step, 0.0, x)
    quant_level = tf.math.round(x / quantization_step)
    x = (quant_level * quantization_step) + min_value
    return x


class QuantizedReLU(tf.keras.layers.Layer):
    """
    ReLU activation that performs fake
    quantization in the float domain.
    
    Quantization gradient is calculated using
    straight-through estimation (STE).
    """
    def __init__(self,
                 num_quantization_levels: int = 256,
                 min_value: float = 0.0,
                 max_value: float = 5.0,
                 cluster_idx: int = 0,
                 **kwargs):
        super(QuantizedReLU, self).__init__(**kwargs)
        self.num_quantization_levels = num_quantization_levels
        self.min_value = min_value
        self.max_value = max_value
        self.cluster_idx = cluster_idx

        # NOTE: normally denominator would have "num_quantization_levels - 1" here,
        #       however in case of ReLU layers we want 257 values because we exclude 0
        #       from the cached table.
        self.quantization_step = (max_value - min_value) / num_quantization_levels

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_quantization_levels': self.num_quantization_levels,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'cluster_idx': self.cluster_idx,
        })
        
        return config

    @tf.custom_gradient
    def quantize(self, x):
        x = quant_dequant_float(x, self.min_value, self.max_value, self.quantization_step, is_relu=True)

        def grad(upstream):
            return upstream 
        return x, grad
    
    def call(self, x):
        y = self.quantize(tf.nn.relu(x))
        return y


class QuantizedDense(tf.keras.layers.Dense):
    """
    A dense layer that performs fake quantization
    of weights but not biases in the float domain.

    Quantization gradient is calculated using
    straight-through estimation (STE).
    """
    def __init__(self,
                 num_quantization_levels: int = 256,
                 min_w_value: float = -1.0,
                 max_w_value: float = 1.0,
                 cluster_idx: int = 0,
                 *args,
                 **kwargs):
        super(QuantizedDense, self).__init__(*args, **kwargs)
        self.num_quantization_levels = num_quantization_levels
        
        self.min_w_value = min_w_value
        self.max_w_value = max_w_value
        self.cluster_idx = cluster_idx

        self.w_quantization_step = (max_w_value - min_w_value) / (num_quantization_levels - 1)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_quantization_levels': self.num_quantization_levels,
            'min_w_value': self.min_w_value,
            'max_w_value': self.max_w_value,
            'cluster_idx': self.cluster_idx,
        })
        
        return config

    @tf.custom_gradient
    def quantize_w(self, w):
        w = quant_dequant_float(w, self.min_w_value, self.max_w_value, self.w_quantization_step, is_relu=False)

        def grad(upstream):
            return upstream 
        return w, grad
    
    def call(self, inputs):
        outputs = tf.matmul(inputs, self.quantize_w(self.kernel))

        if self.use_bias: 
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class QuantizedConv2D(tf.keras.layers.Conv2D):
    """
    A Conv2D layer that performs fake quantization
    of weights but not biases in the float domain.

    Quantization gradient is calculated using
    straight-through estimation (STE).
    """
    def __init__(self,
                 num_quantization_levels: int = 256,
                 min_w_value: float = -1.0,
                 max_w_value: float = 1.0,
                 cluster_idx: int = 0,
                 *args,
                 **kwargs):
        super(QuantizedConv2D, self).__init__(*args, **kwargs)
        self.num_quantization_levels = num_quantization_levels
        
        self.min_w_value = min_w_value
        self.max_w_value = max_w_value
        self.cluster_idx = cluster_idx

        self.w_quantization_step = (max_w_value - min_w_value) / (num_quantization_levels - 1)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_quantization_levels': self.num_quantization_levels,
            'min_w_value': self.min_w_value,
            'max_w_value': self.max_w_value,
            'cluster_idx': self.cluster_idx,
        })

        return config

    @tf.custom_gradient
    def quantize_w(self, w):
        w = quant_dequant_float(w, self.min_w_value, self.max_w_value, self.w_quantization_step, is_relu=False)

        def grad(upstream):
            return upstream 
        return w, grad
    
    def call(self, inputs):
        """
        Call method is copied and modified from Conv() base class
        to perform quant-dequant of weights and biases before computations.
        """
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        q_w = self.quantize_w(self.kernel)
        if self.groups > 1:
            outputs = self._jit_compiled_convolution_op(
                inputs, tf.convert_to_tensor(q_w)
            )
        else:
            outputs = self.convolution_op(inputs, q_w)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(
                            o, self.bias, data_format=self._tf_data_format
                        )

                    outputs = conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1
                    )
                else:
                    outputs = tf.nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format
                    )

        if not tf.executing_eagerly() and input_shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs


class QuantizedInput(tf.keras.layers.Layer):
    """
    Input layer that performs fake quantization in the float domain
    of input and returns it.
    """
    def __init__(self,
                 num_quantization_levels: int = 256,
                 min_value: float = -0.5,
                 max_value: float = 0.5,
                 **kwargs):
        super(QuantizedInput, self).__init__(**kwargs)
        self.num_quantization_levels = num_quantization_levels
        self.min_value = min_value
        self.max_value = max_value

        self.quantization_step = (max_value - min_value) / (num_quantization_levels - 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_quantization_levels': self.num_quantization_levels,
            'min_value': self.min_value,
            'max_value': self.max_value,
        })
        
        return config

    @tf.custom_gradient
    def quantize(self, x):
        x = quant_dequant_float(x, self.min_value, self.max_value, self.quantization_step, is_relu=False)

        def grad(upstream):
            return upstream 
        return x, grad
    
    def call(self, x):
        y = self.quantize(x)
        return y


class BinarySearchSNNDense(tf.keras.layers.Layer):
    """
    A spiking dense layer that instead of performing multiplications
    of activations with weights performs binary search of cached
    multiplication results through SNN neurons. Bias is added in a standard way.

    Number of time steps required for forward pass of this layer
    is equal to 2 * bits, where bits encode number of quantization
    levels for activations (e.g. 8 bits encode 256 values which result
    in simulation steps of 16).

    Actual number of values that the binary search operates on is equal to 2**bits + 1
    as zero is also included but it doesn't reserve additional memory in cached products
    table fro activations (these have only 256 possible activation values in the table and
    not 257, because if activation is 0 then it's product with any weight will also be 0).

    Args:
    - w_map: array of shape Win x Wout containing indices of weights
             in cached_products array.
    - bias: array of shape Wout, containing float32 biases of the original network.
    - bits: number of bits indicating the number of quantization levels for activations
    - is_relu: if True min_activ_value is shifted to omit including 0 in cached products table (shift occurs offline).
               This option works only if ReLU was quantized using 257 quantization levels including 0
               with an interval of <0, max_activ_val>. Should be set to False for QuantizedInput layer conversion. 
    """
    def __init__(self,
                 w_map: np.ndarray,
                 min_activ_value: float,
                 max_activ_value: float,
                 bits: int,
                 bias: Optional[np.ndarray] = None,
                 is_relu: bool = True,
                 *args,
                 **kwargs):
        super(BinarySearchSNNDense, self).__init__(*args, **kwargs)
        self.w_map = w_map
        self.min_activ_value = min_activ_value
        self.max_activ_value = max_activ_value
        self.bits = bits
        self.bias = bias
        self.is_relu = is_relu

        max_min_diff = max_activ_value - min_activ_value
        if is_relu:
            self.q_step = max_min_diff / 2**bits
            self.min_activ_value += self.q_step
        else:
            self.q_step = max_min_diff / (2**bits - 1)

        # Pre-compute index offsets, interval_halves and v_init_offsets
        quant_levels = 2**bits
        idx_div = quant_levels
        self.index_offsets = []
        for _ in range(bits):
            idx_div /= 2
            self.index_offsets.append(int(idx_div))
        
        self.interval_halves = []
        for b in range(bits, 0, -1):
            self.interval_halves.append((2**b - 1) * self.q_step / 2.0)

        self.v_init_offsets = []
        for int_h in self.interval_halves[:-1]:
            self.v_init_offsets.append(int_h + self.interval_halves[-1])

    def get_config(self):
        config = super().get_config()
        config.update({
            'w_map': self.w_map,
            'min_activ_value': self.min_activ_value,
            'max_activ_value': self.max_activ_value,
            'bits': self.bits,
            'bias': self.bias,
            'is_relu': self.is_relu,
        })
        
        return config

    def call(self,
             input_tensor,
             cached_products):
        """
        Args:
        - input_tensor: input activations tensor of shape N x Win
        - cached_products: cached products of activations and weights of the next layer,
                           of shape: activ_q_levels x w_q_levels
        """
        if self.is_relu:
            inp_zero_mask = input_tensor <= self.min_activ_value

        activ_map = perform_binary_search(input_tensor, self.bits, self.min_activ_value,
                                          self.interval_halves, self.v_init_offsets, self.index_offsets)
        batch_size = tf.shape(input_tensor)[0]
        w_shape = tf.shape(self.w_map)
        cached_idxs = tf.concat([tf.tile(activ_map[:, :, tf.newaxis], [1, 1, w_shape[1]])[..., tf.newaxis],
                                 tf.tile(self.w_map[tf.newaxis, :, :], [batch_size, 1, 1])[..., tf.newaxis]],
                                axis=-1)
        
        cached_idxs = tf.reshape(cached_idxs, [-1, 2])
        cached_vals = tf.gather_nd(cached_products, cached_idxs)
        
        if self.is_relu:
            inp_zero_mask = tf.tile(inp_zero_mask[:, :, tf.newaxis],
                                    [1, 1, w_shape[1]])
            inp_zero_mask = tf.reshape(inp_zero_mask, [-1])
            cached_vals = tf.where(inp_zero_mask, 0.0, cached_vals)
        
        cached_vals = tf.reshape(cached_vals, [batch_size, w_shape[0], w_shape[1]])
        output_tensor = tf.reduce_sum(cached_vals, axis=-2)
        if self.bias is not None:
            output_tensor += self.bias

        return output_tensor


def perform_binary_search(input_tensor: tf.Tensor,
                          bits: int,
                          min_val: float,
                          interval_halves: List[float],
                          v_init_offsets: List[float],
                          index_offsets: List[int]):
    """
    This function performs element-wise binary search algorithm by querying SNN layer
    repeatedly over two time steps. The number of necessary time steps is equal to
    two times number of bits used to compute the number of quantization levels (8 bits means 256 levels, 9 means 512, etc.).
    For example 256 quantization levels require SNN query over 16 time steps total.

    Returns tensor of index offsets indicating quantization levels for each value in input tensor.

    NOTE: Function performs solely multiplications of constants so
          they can be optimized away enabling forward pass of the
          network without any multiplications.
    """
    v_thresh = input_tensor
    idx_offsets = tf.zeros(tf.shape(input_tensor), dtype=tf.int32)
    v_init = tf.ones_like(input_tensor) * min_val
    for b in range(bits):
        v_state = tf.identity(v_init)
        v_add = tf.ones_like(input_tensor) * interval_halves[b]
        spike_times = tf.ones_like(input_tensor) * -1.0
        for s in tf.range(2, dtype=tf.float32):
            v_state += v_add
            spike_times = tf.where((v_state >= v_thresh) & (spike_times < 0.0),
                                    s,
                                    spike_times)

        increase_mask = spike_times != 0.0
        idx_offsets = tf.where(increase_mask,
                               idx_offsets + index_offsets[b],
                               idx_offsets)
        if b != bits - 1:
            v_init = tf.where(increase_mask,
                              v_init + v_init_offsets[b],
                              v_init)

    return idx_offsets
