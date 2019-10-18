from tensorflow import keras

from video_util import frame_gen
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
    callbacks = []
    model = keras.models.load_model('./models/sports1m-full-compiled.h5')
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # train_generator = train_datagen.flow_from_directory(
    #     os.path.join('data', 'train'),
    #     target_size=(299, 299),
    #     batch_size=32,
    #     classes=data.classes,
    #     class_mode='categorical')
    #
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=100,
    #     validation_data=validation_generator,
    #     validation_steps=10,
    #     epochs=nb_epoch,
    #     callbacks=callbacks
    # )

    # with open('labels.txt', 'r') as f:
    #     labels = [line.strip() for line in f.readlines()]
    # print('Total labels: {}'.format(len(labels)))

    IMG_HEIGHT = 171
    IMG_WIDTH = 128

    vidstream = frame_gen('./videos/curling.mp4', IMG_WIDTH, IMG_HEIGHT)
    for batchseq in vidstream:
        X = vid[0:(0 + FRAME_BATCH_LEN), :, :, :]


if __name__ == '__main__':
    main()
