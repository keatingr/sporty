import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import sleep
from settings import FRAME_BATCH_LEN
from video_util import frame_gen
import copy
from c3d_model import build_model


def diagnose(data, verbose=True, label='input', plots=False):
    # Convolution3D?
    backend='tf'
    if data.ndim > 2:
        if backend == 'th':
            data = np.transpose(data, (1, 2, 3, 0))
        # else:
        #     data = np.transpose(data, (0, 2, 1, 3))
        min_num_spatial_axes = 10
        max_outputs_to_show = 3
        ndim = data.ndim
        print("[Info] {}.ndim={}".format(label, ndim))
        print("[Info] {}.shape={}".format(label, data.shape))
        for d in range(ndim):
            num_this_dim = data.shape[d]
            if num_this_dim >= min_num_spatial_axes:  # check for spatial axes
                # just first, center, last indices
                range_this_dim = [0, num_this_dim // 2, num_this_dim - 1]
            else:
                # sweep all indices for non-spatial axes
                range_this_dim = range(num_this_dim)
            for i in range_this_dim:
                new_dim = tuple([d] + list(range(d)) + list(range(d + 1, ndim)))
                sliced = np.transpose(data, new_dim)[i, ...]
                print("[Info] {}, dim:{} {}-th slice: "
                      "(min, max, mean, std)=({}, {}, {}, {})".format(
                    label,
                    d, i,
                    np.min(sliced),
                    np.max(sliced),
                    np.mean(sliced),
                    np.std(sliced)))
        if plots:
            # assume (l, h, w, c)-shaped input
            if data.ndim != 4:
                print("[Error] data (shape={}) is not 4-dim. Check data".format(
                    data.shape))
                return
            l, h, w, c = data.shape
            if l >= min_num_spatial_axes or \
                    h < min_num_spatial_axes or \
                    w < min_num_spatial_axes:
                print("[Error] data (shape={}) does not look like in (l,h,w,c) "
                      "format. Do reshape/transpose.".format(data.shape))
                return
            nrows = int(np.ceil(np.sqrt(data.shape[0])))
            # BGR
            if c == 3:
                for i in range(l):
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.subplot(nrows, nrows, i + 1)  # doh, one-based!
                    im = np.squeeze(data[i, ...]).astype(np.float32)
                    im = im[:, :, ::-1]  # BGR to RGB
                    # force it to range [0,1]
                    im_min, im_max = im.min(), im.max()
                    if im_max > im_min:
                        im_std = (im - im_min) / (im_max - im_min)
                    else:
                        print("[Warning] image is constant!")
                        im_std = np.zeros_like(im)
                    plt.imshow(im_std)
                    plt.axis('off')
                    plt.title("{}: t={}".format(label, i))
                plt.show()
                # plt.waitforbuttonpress()
            else:
                for j in range(min(c, max_outputs_to_show)):
                    for i in range(l):
                        mng = plt.get_current_fig_manager()
                        mng.resize(*mng.window.maxsize())
                        plt.subplot(nrows, nrows, i + 1)  # doh, one-based!
                        im = np.squeeze(data[i, ...]).astype(np.float32)
                        im = im[:, :, j]
                        # force it to range [0,1]
                        im_min, im_max = im.min(), im.max()
                        if im_max > im_min:
                            im_std = (im - im_min) / (im_max - im_min)
                        else:
                            print("[Warning] image is constant!")
                            im_std = np.zeros_like(im)
                        plt.imshow(im_std)
                        plt.axis('off')
                        plt.title("{}: o={}, t={}".format(label, j, i))
                    plt.show()
                    # plt.waitforbuttonpress()
    elif data.ndim == 1:
        print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
            label,
            np.min(data),
            np.max(data),
            np.mean(data),
            np.std(data)))
        print("[Info] data[:10]={}".format(data[:10]))

    return


def main():
    IMG_HEIGHT = 171  # 171
    IMG_WIDTH = 128  # 128
    START_FRAME = 1
    video_file = './videos/disc_golf.mp4'

    model = tf.keras.models.load_model('./models/sports1m-keras-tf2.h5')

    model.compile(loss='mean_squared_error', optimizer='sgd')

    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

    cap = cv2.VideoCapture(video_file)
    if not cap:
        print("No video loaded {}".format(video_file))
        exit()
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:  # end of stream
            break
        img = np.array(img, dtype=np.float32)
        vid.append(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)))
        if len(vid) == FRAME_BATCH_LEN:
            X = vid
            X -= mean_cube  # TODO mean avg is very important!
            X = X[:, 8:120, 30:142, :]  # (l, h, w, c)  # TODO center crop is very important! try without it!
            p = model.predict(np.array([X]))  # TODO can just use X?
            p_label = '{:.5f} - {}'.format(max(p[0]), labels[int(np.argmax(p[0]))])
            img = img / 255
            cv2.putText(img, p_label, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 75, 0), 1)
            vid.pop(0)
        cv2.imshow('Sporty', img)
        cv2.moveWindow('Sporty', 300, 150)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #
    #     output = model.predict(np.array([X]))  # TODO difference from predict_on_batch?
    #     for idx in batchseq:
    #         p_label = '{:.5f} - {}'.format(max(output[0]), labels[int(np.argmax(output[0]))])
    #         img = idx/256
    #         cv2.putText(img, p_label, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 75, 0), 1)
    #         cv2.imshow('', img)
    #         sleep(.05)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

        # output = model.predict_on_batch(np.array([predict_input]))  # why not /256 TODO major test dividing by 256
        #
        # # show results
        # # print('Saving class probabilitities in probabilities.png')
        # # plt.plot(output[0])
        # # plt.title('Probability')
        # # plt.savefig("probabilities.png")
        # print('Position of maximum probability: {}'.format(tf.math.argmax(output[0])))
        # print('Maximum probability: {:.5f}'.format(max(output[0])))
        # print('Corresponding label: {}'.format(labels[tf.math.argmax(output[0])]))
        #
        # top_inds = tf.argsort(output[0])[::-1][:5]  # reverse sort and take five largest items
        # print('\n{} - Top 5 probabilities and labels:'.format(k))  # TODO show frame not iter/k
        # for i in top_inds:
        #     print('{1}: {0:.5f}'.format(output[0][i], labels[i]))


if __name__ == '__main__':
    main()
