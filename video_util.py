import cv2
import numpy as np
from settings import FRAME_BATCH_LEN


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
                frameset = np.array(vid, dtype=np.float32)  # TODO RESEARCH use tensorflow?
                yield np.array([frameset])
                vid = []

    except:
        print("Problem reading from video {}".format(filename))