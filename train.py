from tensorflow import keras
from c3d_model import build_model
from video_util import train_frame_gen
from settings import FRAME_BATCH_LEN


def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


def main():
    IMG_HEIGHT = 112
    IMG_WIDTH = 112

    # model = keras.models.load_model('./models/sports1m-keras-tf2.h5')
    model = build_model()

    model.compile(loss='mean_squared_error', optimizer='sgd')

    model.summary()
    # with open('labels.txt', 'r') as f:
    #     labels = [line.strip() for line in f.readlines()]
    # print('Total labels: {}'.format(len(labels)))

    # TODO center crop within the frame gen prob see also predict code
    train_stream_cubes = train_frame_gen('./videos/nyc_driving.mp4', img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
    val_stream_cubes = train_frame_gen('./videos/nyc_driving.mp4', img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

    # TODO batch into 8/16/32
    # for batchseq in vidstream:
    #     X = vid[0:(0 + FRAME_BATCH_LEN), :, :, :]

    callbacks = []

    history = model.fit_generator(  # tuple (inputs, targets) or (inputs, targets, sample_weights) https://www.tensorflow.org/api_docs/python/tf/keras/Model
        train_stream_cubes,
        steps_per_epoch=100,
        validation_data=val_stream_cubes,
        validation_steps=15,
        epochs=1,
        callbacks=callbacks
    )
    model.save('test.h5')


if __name__ == '__main__':
    main()
