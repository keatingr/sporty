import cv2
import numpy as np
import tensorflow as tf
from time import sleep
from settings import FRAME_BATCH_LEN
import copy


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


def main():
    """
    Pipeline video frames to inference nodes. Eventually this could get too expensive and would rely on shipping references
    to shared memory/disk instead of the entire tensors; for now ship the whole thing
    :return:
    """
    IMG_HEIGHT = 171  # 171
    IMG_WIDTH = 128  # 128
    START_FRAME = 1
    video_file = './videos/basketball.mp4'

    model = tf.keras.models.load_model('./models/sports1m-keras-tf2.h5')
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # convert video into python generator of frames
    vidstream = frame_gen(video_file, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, start_frame=START_FRAME)

    for k, batchseq in enumerate(vidstream):

        raw_frame = batchseq[0:(0 + FRAME_BATCH_LEN), :, :, :]
        raw_frame = copy.deepcopy(raw_frame)
        X = copy.deepcopy(raw_frame)
        # TODO tag with sequence (frame index) for network/performance issues
        # TODO stream this data into the channel

        # TODO subscriber inference node reads frame and runs through its model
        predict_input = X[:, 8:120, 30:142, :]
        output = model.predict_on_batch(np.array([predict_input]))

        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print('Total labels: {}'.format(len(labels)))
        p_label = '{:.5f} - {}'.format(max(output[0]), labels[int(np.argmax(output[0]))])

        # TODO inference node (voter node) puts tensor and prediction into response channel along with inference node identifier

        # TODO a separate node that subscribes to response channel gathers all the responses from prediction nodes and tallies the votes and makes a final prediction and sends it with some debug metadata to the web app

        # TODO web app (for now a local python console that uses imshow) displays the streaming video with predictions and also provides real-time "debug/inference decision" information in a TBD useful format
        img = raw_frame[0]/256
        cv2.putText(img, p_label, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 75, 0), 1)
        cv2.imshow('', img)
        sleep(.25)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
