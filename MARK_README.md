1) tf2 won't load the model file or something so using tf 1.15 for now
2) refactored tf.argmax to use np.argmax

Models:
1) conv3d_deepnetA_sport1m_iter_1900000 - sports1m pretrained downloaded from get_weights_and_mean.sh
2) sports1m-...keras-tf-1.15.h5 - older keras/tf versions
3) sports1m-keras-tf2 - tf2/keras compat

Not sure how I converted whatever the originals were to keras