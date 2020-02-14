import cv2
import numpy as np
from settings import FRAME_BATCH_LEN
import tensorflow as tf


def frame_gen(filename, img_width=0, img_height=0, start_frame=0):
    """
    Image generator for video frames. Treats all videos as just bags of FRAME_BATCH_LEN,
    concatenating videos to each other in the stream.
    Note: drops final frames not part of a full batch of FRAME_BATCH_LEN; frames dropped when not divisible evenly
    :param filename: (str) relative or full path to video file, including extension inputvideo.mp4
    :return: (ndarray or None) Continguous batch sequences of length FRAME_BATCH_LEN otherwise None
    """
    try:
        cap = cv2.VideoCapture(filename)
        if not cap:
            print("No video loaded {}".format(filename))
            return
        idx = 0
        vid = []
        while True:
            ret, img = cap.read()
            if not ret:  # end of stream
                return
            vid.append(cv2.resize(img, (img_height, img_width)))
            idx += 1
            if idx % FRAME_BATCH_LEN == 0:
                yield np.array(vid, dtype=np.float32)  # TODO RESEARCH use tensorflow?
                vid = []

    except:
        print("Problem reading from video {}".format(filename))


def get_class_one_hot(self, class_str):
    """Given a class as a string, return its number in the classes
    list. This lets us encode and one-hot it for training."""
    # Encode it first.
    label_encoded = self.classes.index(class_str)

    # Now one-hot it.
    label_hot = tf.one_hot(label_encoded, len(self.classes))  # TODO verify this transcribing old keras to_categorical

    assert len(label_hot) == len(self.classes)

    return label_hot

def train_frame_gen(filename, img_width=0, img_height=0):
    """
    """
    try:
        cap = cv2.VideoCapture(filename)
        if not cap:
            print("No video loaded {}".format(filename))
            return
        idx = 0
        vid = []
        while True:
            ret, img = cap.read()
            if not ret:  # end of stream
                return
            vid.append(cv2.resize(img, (img_height, img_width)))
            # center crop - X = X[:, 8:120, 30:142, :]  # (l, h, w, c)
            idx += 1
            if idx % FRAME_BATCH_LEN == 0:
                frameset = np.array(vid, dtype=np.float32)  # TODO RESEARCH use tensorflow? also keras.utils.Sequence lets you multithread
                # labels = [0] * 486
                # labels.insert(0, 1)
                y = tf.one_hot([500], 487) # TODO make actual category
                output = np.array([frameset]), np.array(y)  # tf.keras fit_generator expexts tuple
                yield output
                vid = []

    except:
        print("Problem reading from video {}".format(filename))
