import tensorflow as tf
from tensorflow import keras


def build_model():
    input_shape = (16, 112, 112, 3)  # l, h, w, c

    model = keras.models.Sequential([
        keras.layers.Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', input_shape=input_shape),
        keras.layers.MaxPooling3D(pool_size =(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1'),
        keras.layers.Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2'),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2'),
        keras.layers.Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a'),
        keras.layers.Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b'),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3'),
        keras.layers.Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a'),
        keras.layers.Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b'),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4'),
        keras.layers.Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5a'),
        keras.layers.Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5b'),
        keras.layers.ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool5'),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu', name='fc6'),
        keras.layers.Dropout(.5),
        keras.layers.Dense(4096, activation='relu', name='fc7'),
        keras.layers.Dropout(.5),
        keras.layers.Dense(487, activation='softmax', name='fc8')
    ])
    return model


# def get_int_model(model, layer):
#     input_shape = (16, 112, 112, 3)  # l, h, w, c
#     # Theano: input_shape = (3, 16, 112, 112)  # c, l, h, w
#
#     int_model = Sequential()
#
#     int_model.add(Convolution3D(64, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv1',
#                                 input_shape=input_shape,
#                                 weights=model.layers[0].get_weights()))
#     if layer == 'conv1':
#         return int_model
#     int_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
#                                border_mode='valid', name='pool1'))
#     if layer == 'pool1':
#         return int_model
#
#     # 2nd layer group
#     int_model.add(Convolution3D(128, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv2',
#                                 weights=model.layers[2].get_weights()))
#     if layer == 'conv2':
#         return int_model
#     int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                                border_mode='valid', name='pool2'))
#     if layer == 'pool2':
#         return int_model
#
#     # 3rd layer group
#     int_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv3a',
#                                 weights=model.layers[4].get_weights()))
#     if layer == 'conv3a':
#         return int_model
#     int_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv3b',
#                                 weights=model.layers[5].get_weights()))
#     if layer == 'conv3b':
#         return int_model
#     int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                                border_mode='valid', name='pool3'))
#     if layer == 'pool3':
#         return int_model
#
#     # 4th layer group
#     int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv4a',
#                                 weights=model.layers[7].get_weights()))
#     if layer == 'conv4a':
#         return int_model
#     int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv4b',
#                                 weights=model.layers[8].get_weights()))
#     if layer == 'conv4b':
#         return int_model
#     int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                                border_mode='valid', name='pool4'))
#     if layer == 'pool4':
#         return int_model
#
#     # 5th layer group
#     int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv5a',
#                                 weights=model.layers[10].get_weights()))
#     if layer == 'conv5a':
#         return int_model
#     int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
#                                 border_mode='same', name='conv5b',
#                                 weights=model.layers[11].get_weights()))
#     if layer == 'conv5b':
#         return int_model
#     int_model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad'))
#     int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                                border_mode='valid', name='pool5'))
#     if layer == 'pool5':
#         return int_model
#
#     int_model.add(Flatten())
#     # FC layers group
#     int_model.add(Dense(4096, activation='relu', name='fc6',
#                         weights=model.layers[15].get_weights()))
#     if layer == 'fc6':
#         return int_model
#     int_model.add(Dropout(.5))
#     int_model.add(Dense(4096, activation='relu', name='fc7',
#                         weights=model.layers[17].get_weights()))
#     if layer == 'fc7':
#         return int_model
#     int_model.add(Dropout(.5))
#     int_model.add(Dense(487, activation='softmax', name='fc8',
#                         weights=model.layers[19].get_weights()))
#     if layer == 'fc8':
#         return int_model
#
#     return None
