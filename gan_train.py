import tensorflow as tf
import numpy as np
import cv2, os

LATENT_DIM = 100
HEIGHT, WIDTH, DEPTH = 128, 128, 3
GEN_H1, GEN_W1 = 4, 4
GEN_D1, GEN_D2, GEN_D3, GEN_D4, GEN_D5 = 512, 256, 128, 64, 32
DIS_D1, DIS_D2, DIS_D3, DIS_D4 = 64, 128, 256, 512

BATCH_SIZE, DIS_ITERS, LEARNING_RATE, TRAIN_ITERS = 100, 3, 1e-4, 100000
EVAL_ROWS, EVAL_COLS, SAMPLES_PATH, EVAL_INTERVAL = 8, 12, './samples', 500
SAVE_INTERVAL, MODEL_PATH = 10000, './model'

# -------------------------------------------------------------------------
# -------------------------- batch normalization --------------------------
# -------------------------------------------------------------------------

def assign_decay(orig_val, new_val, momentum, name):

	with tf.name_scope(name):

		scaled_diff = (1 - momentum) * (new_val - orig_val)

	return tf.assign_add(orig_val, scaled_diff)

def batch_norm(x, train_logical, decay, epsilon, scope=None, shift=True, scale=False):

	channels = x.get_shape()[-1]
	ndim = len(x.shape)

	with tf.variable_scope(scope):

		moving_m = tf.get_variable('mean', [channels], initializer=tf.zeros_initializer, trainable=False)
		moving_v = tf.get_variable('var', [channels], initializer=tf.ones_initializer, trainable=False)

		if train_logical:

			m, v = tf.nn.moments(x, range(ndim - 1))
			update_m = assign_decay(moving_m, m, decay, 'update_mean')
			update_v = assign_decay(moving_v, v, decay, 'update_var')

			with tf.control_dependencies([update_m, update_v]):
				output = (x - m) * tf.rsqrt(v + epsilon)

		else:
			m, v = moving_m, moving_v
			output = (x - m) * tf.rsqrt(v + epsilon)

		if scale:
			output *= tf.get_variable('gamma', [channels], initializer=tf.ones_initializer)

		if shift:
			output += tf.get_variable('beta', [channels], initializer=tf.zeros_initializer)

	return output

# -------------------------------------------------------------------
# -------------------------- record reader --------------------------
# -------------------------------------------------------------------

def read_example(filename):

	reader = tf.FixedLengthRecordReader(HEIGHT*WIDTH*DEPTH)
	filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
	key, serialized_example = reader.read(filename_queue)

	image_raw = tf.decode_raw(serialized_example, tf.uint8)
	image = tf.cast(tf.reshape(image_raw, [HEIGHT, WIDTH, DEPTH]), tf.float32)

	return image

def preprocess(image):

	image = image/255.0

	image = tf.image.random_brightness(image, max_delta=0.1)
	image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
	image = tf.image.random_flip_left_right(image)

	return image

def make_batch(image, batch_size):

	min_queue_examples = 500

	batch_images = tf.train.shuffle_batch([image], batch_size=batch_size, 
					      capacity=min_queue_examples+10*batch_size,
					      min_after_dequeue=min_queue_examples, num_threads=8)

	return batch_images

def read_record(filename, batch_size):
	
	image = read_example(filename)
	image = preprocess(image)
	batch_images = make_batch(image,batch_size)

	return batch_images

# ---------------------------------------------------------------
# -------------------------- generator -------------------------
# ---------------------------------------------------------------

def conv_transpose(x, out_depth, kernel, strides, name):

	in_shape = x.get_shape().as_list()
	in_batch = tf.shape(x)[0]
	in_height, in_width, in_depth = in_shape[1:]
	out_shape = [in_batch, in_height*strides[0], in_width*strides[1], out_depth]

	with tf.name_scope(name): 

		w = tf.get_variable('w', shape=[kernel[0], kernel[1], out_depth, in_depth], initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable('b', shape=[out_depth], initializer=tf.zeros_initializer())

		conv = tf.nn.conv2d_transpose(x, filter=w, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME', name='deconv')
		conv = tf.add(conv, b, name='add')

	return conv	

def gen_conv_block(x, depth, train_logical, scope, final=False):

	with tf.variable_scope(scope): 

		conv = conv_transpose(x, depth, kernel=[5,5], strides=[2,2], name='conv')

		if final:
			act = tf.nn.sigmoid(conv, name='act')

		else:
			bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay = 0.9, scope='bn')
			act = tf.nn.relu(bn, name='act')

	return act

def gen_fc_block(x, height, width, depth, train_logical, scope):

	latent_dim = x.get_shape()[1]

	with tf.variable_scope(scope):

		w = tf.get_variable('w', shape=[latent_dim, height*width*depth], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable('b', shape=[height * width * depth], dtype=tf.float32, initializer=tf.zeros_initializer())
		flat_conv = tf.add(tf.matmul(x, w), b, name='flat_conv')

		conv = tf.reshape(flat_conv, shape=[-1, height, width, depth], name='conv')
		bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay = 0.9, scope='bn')
		act = tf.nn.relu(bn, name='act')

	return act

# -------------------------------------------------------------------
# -------------------------- discriminator --------------------------
# -------------------------------------------------------------------

def lrelu(x, leak, name):
 
	return tf.maximum(x, leak*x, name=name)

def conv_layer(x, out_depth, kernel, strides, name):

	in_depth = x.get_shape()[3]

	with tf.name_scope(name): 

		w = tf.get_variable('w',shape=[kernel[0], kernel[1], in_depth, out_depth], initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable('b',shape=[out_depth], initializer=tf.zeros_initializer())
		
		conv = tf.nn.conv2d(x, filter=w, strides=[1, strides[0], strides[1], 1], padding="SAME", name='conv')
		conv = tf.add(conv, b, name='add')

	return conv

def dis_conv_block(x, out_depth, train_logical, scope, bn_logical=True):

	with tf.variable_scope(scope): 

		conv=conv_layer(x, out_depth, [5,5], [2,2], name='conv')

		if bn_logical == True:
			bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay = 0.9, scope='bn')
			act = lrelu(bn, leak = 0.2, name='act')

		else:
			act = lrelu(conv, leak = 0.2, name='act')

	return act

def dis_fc_block(x, scope):

	dim = int(np.prod(x.get_shape()[1:]))

	with tf.variable_scope(scope):

		flat = tf.reshape(x, shape=[-1, dim], name='flat')
		w = tf.get_variable('w', shape=[dim, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable('b', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
		y = tf.add(tf.matmul(flat, w), b, name='y')

	return y

# -------------------------------------------------------------
# -------------------------- training --------------------------
# --------------------------------------------------------------

def make_dir(directory):

	if not os.path.exists(directory):
		os.makedirs(directory)

def generator(input, train_logical):

	act1 = gen_fc_block(input, GEN_H1, GEN_W1, GEN_D1, train_logical, 'fc1')
	act2 = gen_conv_block(act1, GEN_D2, train_logical, 'conv2')
	act3 = gen_conv_block(act2, GEN_D3, train_logical, 'conv3')
	act4 = gen_conv_block(act3, GEN_D4, train_logical, 'conv4')
	act5 = gen_conv_block(act4, GEN_D5, train_logical, 'conv5')
	act6 = gen_conv_block(act5, DEPTH, train_logical, 'conv6', final=True)

	return act6

def discriminator(input, train_logical):

	act1 = dis_conv_block(input, DIS_D1, train_logical, 'conv1', bn_logical=False)
	act2 = dis_conv_block(act1, DIS_D2, train_logical, 'conv2')
	act3 = dis_conv_block(act2, DIS_D3, train_logical, 'conv3')
	act4 = dis_conv_block(act3, DIS_D4, train_logical, 'conv4')
	y = dis_fc_block(act4, 'fc5')

	return y

def train():

	latent_input = tf.placeholder(tf.float32, shape=[None, LATENT_DIM], name='latent_input')

	with tf.device('/cpu:0'):
		with tf.name_scope('train_batch'):
			real_image = read_record('./train.record',BATCH_SIZE)

	with tf.variable_scope('gen') as scope:
		with tf.name_scope('train'):
			fake_image = generator(latent_input, train_logical=True)
		scope.reuse_variables()
		with tf.name_scope('eval'):
			fake_image_eval = generator(latent_input, train_logical=False)
  
	with tf.variable_scope('dis') as scope:
		with tf.name_scope('real'):
			real_result = discriminator(real_image, train_logical=True)
		scope.reuse_variables()
		with tf.name_scope('fake'):
			fake_result = discriminator(fake_image, train_logical=True)
    
	with tf.name_scope('loss'):
		dis_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
		gen_loss = -tf.reduce_mean(fake_result)

	with tf.name_scope('optimizer'):
		gan_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		dis_vars = [var for var in gan_vars if 'dis' in var.name]
		gen_vars = [var for var in gan_vars if 'gen' in var.name]
		dis_step = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, name='dis_rmsprop').minimize(dis_loss, var_list=dis_vars)
		gen_step = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, name='gen_rmsprop').minimize(gen_loss, var_list=gen_vars)
	
	with tf.name_scope('dis_clip'):
		dis_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in dis_vars]

	sess = tf.Session()
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for i in range(TRAIN_ITERS):

		train_latent_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, LATENT_DIM]).astype(np.float32)

		for j in range(DIS_ITERS):

			sess.run(dis_clip)
			_, dis_loss_data = sess.run([dis_step, dis_loss], feed_dict={latent_input: train_latent_noise})

		_, gen_loss_data = sess.run([gen_step, gen_loss], feed_dict={latent_input: train_latent_noise})

		print('iter: %d, dis_loss: %f, gen_loss: %f' % (i, dis_loss_data, gen_loss_data))

		if (i+1)%SAVE_INTERVAL == 0:

			make_dir(MODEL_PATH)
			saver.save(sess, MODEL_PATH + '/gan', global_step=i+1)

		if (i+1)%EVAL_INTERVAL == 0:

			make_dir(SAMPLES_PATH)
			eval_latent_noise = np.random.uniform(-1.0, 1.0, size=[EVAL_ROWS*EVAL_COLS, LATENT_DIM]).astype(np.float32)
			data = sess.run(fake_image_eval, feed_dict={latent_input: eval_latent_noise})
			data = np.reshape(data*255, (EVAL_ROWS, EVAL_COLS, HEIGHT, WIDTH, DEPTH))
			data = np.concatenate(np.concatenate(data, 1), 1)
			cv2.imwrite(SAMPLES_PATH +'/sample-iter' + str(i) + '.png',data)

	saver.save(sess, MODEL_PATH + '/gan', global_step=i+1)

	coord.request_stop()
	coord.join(threads)

if __name__ == "__main__":
    train()
