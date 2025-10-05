import tensorflow as tf

from typing import Callable, List


def fc2(num_classes, inp_shape, activation = lambda x: tf.keras.layers.ReLU()(x),
        matmul_layers: List[Callable] = None, batch_norm: bool = True, input_processing_layer=None,
        weights_regularizer=None, activity_regularizer=None):
    """
    Fully connected model: 600-10
    """
    if not isinstance(activation, list):
        activation = [activation]

    inp = tf.keras.Input(shape=inp_shape, dtype=tf.float32)
    
    if input_processing_layer is not None:
        x = input_processing_layer(inp)
    else:
        x = inp

    if matmul_layers is None:
        x = tf.keras.layers.Dense(600, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x)
    else:
        x = matmul_layers[0](600, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x)

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = activation[0](x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)

    if matmul_layers is None:
        out = tf.keras.layers.Dense(num_classes, 'softmax', True, 'glorot_uniform')(x)
    else:
        out = matmul_layers[1](num_classes, 'softmax', True, 'glorot_uniform')(x)

    return tf.keras.Model(inputs=inp, outputs=out, name='FC2')


def lenet5(num_classes, inp_shape, activation = lambda x: tf.keras.layers.ReLU()(x),
           matmul_layers: List[Callable] = None, batch_norm: bool = True, input_processing_layer=None,
           weights_regularizer=None, activity_regularizer=None):
    """
    LenNet5 model with avg pooling replaced by max pooling and tanh replaced by ReLU.
    """
    if not isinstance(activation, list):
        activation = [activation] * 4

    inp = tf.keras.layers.Input(shape=inp_shape, dtype=tf.float32)

    if input_processing_layer is not None:
        x = input_processing_layer(inp)
    else:
        x = inp

    if matmul_layers is None:
        x = tf.keras.layers.Conv2D(6, (5, 5), kernel_initializer='he_uniform',
                                   padding='valid', kernel_regularizer=weights_regularizer)(x)
    else:   
        x = matmul_layers[0](6, (5, 5), kernel_initializer='he_uniform',
                             padding='valid', kernel_regularizer=weights_regularizer)(x)
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    
    x = activation[0](x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    if matmul_layers is None:
        x = tf.keras.layers.Conv2D(16, (5, 5), kernel_initializer='he_uniform',
                                   padding='valid', kernel_regularizer=weights_regularizer)(x)
    else:
        x = matmul_layers[1](16, (5, 5), kernel_initializer='he_uniform',
                             padding='valid', kernel_regularizer=weights_regularizer)(x)


    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = activation[1](x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)
    
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    
    if matmul_layers is None:
        x = tf.keras.layers.Dense(120, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x) 
    else:
        x = matmul_layers[2](120, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x) 
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = activation[2](x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)
    
    if matmul_layers is None:
        x = tf.keras.layers.Dense(84, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x) 
    else:
        x = matmul_layers[3](84, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x) 

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = activation[3](x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)
    
    if matmul_layers is None:
        out = tf.keras.layers.Dense(num_classes, 'softmax', True, 'glorot_uniform')(x)
    else:
        out = matmul_layers[4](num_classes, 'softmax', True, 'glorot_uniform')(x)

    return tf.keras.Model(inputs=inp, outputs=out, name='LeNet5')


def vgg_conv(x, n_filters, activation, matmul_layer, batch_norm = True,
             weights_regularizer=None, activity_regularizer=None):
    x = matmul_layer(n_filters, (3, 3), kernel_initializer='he_uniform',
                     padding='same', kernel_regularizer=weights_regularizer)(x)
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = activation(x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)

    return x


def vgg16(num_classes, inp_shape, activation = lambda x: tf.keras.layers.ReLU()(x),
          matmul_layers: List[Callable] = None, batch_norm: bool = True, input_processing_layer=None,
          weights_regularizer=None, activity_regularizer=None, dropout=False):
    """
    Classic VGG16 model.
    """
    if not isinstance(activation, list):
        activation = [activation] * 15

    if matmul_layers is None:
        matmul_layers = [tf.keras.layers.Conv2D for _ in range(13)]
        matmul_layers.extend([tf.keras.layers.Dense for _ in range(3)])

    inp = tf.keras.layers.Input(shape=inp_shape, dtype=tf.float32)
    
    if input_processing_layer is not None:
        x = input_processing_layer(inp)
    else:
        x = inp

    x = vgg_conv(x, 64, activation[0], matmul_layers[0], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 64, activation[1], matmul_layers[1], batch_norm, weights_regularizer, activity_regularizer)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = vgg_conv(x, 128, activation[2], matmul_layers[2], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 128, activation[3], matmul_layers[3], batch_norm, weights_regularizer, activity_regularizer)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = vgg_conv(x, 256, activation[4], matmul_layers[4], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 256, activation[5], matmul_layers[5], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 256, activation[6], matmul_layers[6], batch_norm, weights_regularizer, activity_regularizer)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = vgg_conv(x, 512, activation[7], matmul_layers[7], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 512, activation[8], matmul_layers[8], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 512, activation[9], matmul_layers[9], batch_norm, weights_regularizer, activity_regularizer)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = vgg_conv(x, 512, activation[10], matmul_layers[10], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 512, activation[11], matmul_layers[11], batch_norm, weights_regularizer, activity_regularizer)
    x = vgg_conv(x, 512, activation[12], matmul_layers[12], batch_norm, weights_regularizer, activity_regularizer)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    
    x = matmul_layers[13](4096, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x)

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = activation[13](x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)
    
    if dropout:
        x = tf.keras.layers.Dropout(0.4)(x)
    
    x = matmul_layers[14](4096, None, True, 'he_uniform', kernel_regularizer=weights_regularizer)(x) 

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = activation[14](x)
    if activity_regularizer is not None:
        x = activity_regularizer(x)
    
    if dropout:
        x = tf.keras.layers.Dropout(0.4)(x)

    out = matmul_layers[15](num_classes, 'softmax', True, 'glorot_uniform')(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name='VGG16')
    
    return model


def conv_skip_block(input_tensor, matmul_layers, activations, kernel_sizes,
                    filters, strides, kernel_initializer='he_normal', kernel_regularizer=None,
                    skip_conn_conv=False, batch_norm = True, use_concat = False,
                    skip_conv_layer=tf.keras.layers.Conv2D):
    x = input_tensor
    last_idx = len(filters) - 1
    for c, conv_info in enumerate(zip(filters, kernel_sizes, strides, matmul_layers, activations)):
        nf, ks, str, layer, activ = conv_info
        x = layer(nf, ks, str, padding='same',
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        if c != last_idx:
            x = activ(x)

    xs = input_tensor
    if skip_conn_conv:
        xs = skip_conv_layer(filters[-1], (1, 1), strides[0], padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)(xs)
        if batch_norm:
            xs = tf.keras.layers.BatchNormalization()(xs)

    if not use_concat:
        x = tf.keras.layers.Add()([x, xs])
    else:
        x = tf.keras.layers.Concatenate(axis=-1)([x, xs])

    x = activations[-1](x)

    return x


def resnet32(n_classes, inp_shape, activation = lambda x: tf.keras.layers.ReLU()(x),
             matmul_layers: List[Callable] = None, input_processing_layer=None,
             batch_norm=True, dropout=False):
    if not isinstance(activation, list):
        activation = [activation] * 29

    if matmul_layers is None:
        matmul_layers = [tf.keras.layers.Conv2D for _ in range(31)]
        matmul_layers.extend([tf.keras.layers.Dense for _ in range(1)])
    
    inp = tf.keras.layers.Input(shape=inp_shape, dtype=tf.float32)

    if input_processing_layer is not None:
        x = input_processing_layer(inp)
    else:
        x = inp
    
    x = matmul_layers[0](64, (7, 7), strides=2, padding='valid',
                         kernel_initializer='he_normal')(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = activation[0](x) 
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    del matmul_layers[0]
    del activation[0]

    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [64, 64, 256], [2, 1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[3])
    del matmul_layers[0:4]
    del activation[0:3]
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [64, 64, 256], [1, 1, 1], batch_norm=batch_norm)
    del matmul_layers[0:3]
    del activation[0:3]
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [64, 64, 256], [1, 1, 1], batch_norm=batch_norm)
    del matmul_layers[0:3]
    del activation[0:3]
    
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [128, 128, 512], [2, 1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[3])
    del matmul_layers[0:4]
    del activation[0:3]
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [128, 128, 512], [1, 1, 1], batch_norm=batch_norm)
    del matmul_layers[0:3]
    del activation[0:3]
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [128, 128, 512], [1, 1, 1], batch_norm=batch_norm)
    del matmul_layers[0:3]
    del activation[0:3]

    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [256, 256, 1024], [2, 1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[3])
    del matmul_layers[0:4]
    del activation[0:3]
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [256, 256, 1024], [1, 1, 1], batch_norm=batch_norm)
    del matmul_layers[0:3]
    del activation[0:3]
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(1, 1), (3, 3), (1, 1)], [256, 256, 1024], [1, 1, 1], batch_norm=batch_norm)
    del matmul_layers[0:3]
    del activation[0:3]

    if dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.GlobalMaxPool2D()(x)

    out = matmul_layers[-1](n_classes, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    model = tf.keras.Model(inp, out, name='resnet33')

    return model


def unet5(num_classes, inp_shape, activation = lambda x: tf.keras.layers.ReLU()(x),
          matmul_layers: List[Callable] = None, input_processing_layer=None,
          batch_norm=True):
    if not isinstance(activation, list):
        activation = [activation] * 22

    if matmul_layers is None:
        matmul_layers = [tf.keras.layers.Conv2D for _ in range(32)]
    
    inp = tf.keras.layers.Input(shape=inp_shape, dtype=tf.float32)

    if input_processing_layer is not None:
        x = input_processing_layer(inp)
    else:
        x = inp


    # Contracting path

    # 1
    kernel_regularizer = tf.keras.regularizers.L2(0.0001)
    x1 = conv_skip_block(x, matmul_layers[:2], activation[:2], [(3, 3), (3, 3)], [32, 32], [1, 1], batch_norm=batch_norm,
                         skip_conn_conv=True, skip_conv_layer=matmul_layers[2], kernel_regularizer=kernel_regularizer)
    del matmul_layers[0:3]
    del activation[0:2]

    x = tf.keras.layers.MaxPool2D((2, 2))(x1)

    # 2
    x2 = conv_skip_block(x, matmul_layers[:2], activation[:2], [(3, 3), (3, 3)], [64, 64], [1, 1], batch_norm=batch_norm,
                         skip_conn_conv=True, skip_conv_layer=matmul_layers[2])
    del matmul_layers[0:3]
    del activation[0:2]

    x = tf.keras.layers.MaxPool2D((2, 2))(x2)

    # 3
    x3 = conv_skip_block(x, matmul_layers[:2], activation[:2], [(3, 3), (3, 3)], [128, 128], [1, 1], batch_norm=batch_norm,
                         skip_conn_conv=True, skip_conv_layer=matmul_layers[2])
    del matmul_layers[0:3]
    del activation[0:2]

    x = tf.keras.layers.MaxPool2D((2, 2))(x3)

    # 4
    x4 = conv_skip_block(x, matmul_layers[:3], activation[:3], [(3, 3), (3, 3), (3, 3)], [256, 256, 256], [1, 1, 1], batch_norm=batch_norm,
                         skip_conn_conv=True, skip_conv_layer=matmul_layers[3])
    del matmul_layers[0:4]
    del activation[0:3]

    x = tf.keras.layers.MaxPool2D((2, 2))(x4)

    # Middle
    x = conv_skip_block(x, matmul_layers[:4], activation[:4], [(3, 3), (3, 3), (3, 3), (3, 3)], [512, 512, 512, 512], [1, 1, 1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[4])
    del matmul_layers[0:5]
    del activation[0:4]
    
    # Expanding path

    # 4
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    
    x = tf.keras.layers.Concatenate(axis=-1)([x, x4])
    x = conv_skip_block(x, matmul_layers[:3], activation[:3], [(3, 3), (3, 3), (3, 3)], [256, 256, 256], [1, 1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[3])
    del matmul_layers[0:4]
    del activation[0:3]

    # 3
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    
    x = tf.keras.layers.Concatenate(axis=-1)([x, x3])
    x = conv_skip_block(x, matmul_layers[:2], activation[:2], [(3, 3), (3, 3)], [128, 128], [1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[2])
    del matmul_layers[0:3]
    del activation[0:2]

    # 2
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    
    x = tf.keras.layers.Concatenate(axis=-1)([x, x2])
    x = conv_skip_block(x, matmul_layers[:2], activation[:2], [(3, 3), (3, 3)], [64, 64], [1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[2])
    del matmul_layers[0:3]
    del activation[0:2]

    # 1
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    
    x = tf.keras.layers.Concatenate(axis=-1)([x, x1])
    x = conv_skip_block(x, matmul_layers[:2], activation[:2], [(3, 3), (3, 3)], [32, 32], [1, 1], batch_norm=batch_norm,
                        skip_conn_conv=True, skip_conv_layer=matmul_layers[2])
    del matmul_layers[0:3]
    del activation[0:2]
    
    out = matmul_layers[-1](num_classes, (1, 1), activation='softmax', kernel_initializer='glorot_uniform')(x)
    
    return tf.keras.Model(inp, out, name='unet5')
