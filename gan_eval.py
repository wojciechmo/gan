import tensorflow as tf
import numpy as np
import cv2

LATENT_DIM = 100
ROWS, COLS = 5, 10
HEIGHT, WIDTH, CHANNEL = 128, 128, 3

with tf.Session() as sess: 

	saver = tf.train.import_meta_graph('./model/gan-100000.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./model'))

	graph = tf.get_default_graph()
	latent_input = graph.get_tensor_by_name('latent_input:0')
	image_eval = graph.get_tensor_by_name('gen/eval/conv6/act:0')

	eval_latent_noise = np.random.uniform(-1.0, 1.0, size=[ROWS*COLS, LATENT_DIM]).astype(np.float32)

	data = sess.run(image_eval, feed_dict = {latent_input: eval_latent_noise})
	data = np.reshape(data, (ROWS, COLS, HEIGHT, WIDTH, CHANNEL))
	data = np.concatenate(np.concatenate(data, 1), 1)
	cv2.imshow('eval_img', data)
	cv2.waitKey()
