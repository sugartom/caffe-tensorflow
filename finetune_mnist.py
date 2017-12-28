# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mynet import LeNet as MyNet



# Yitao-TLS-Begin
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

# tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            # 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 32

def gen_data(source):
    while True:
        indices = range(len(source.images))
        random.shuffle(indices)
        for i in indices:
            image = np.reshape(source.images[i], (28, 28, 1))
            label = source.labels[i]
            yield image, label

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        label_batch = []
        for _ in range(batch_size):
            image, label = next(data_gen)
            image_batch.append(image)
            label_batch.append(label)
        yield np.array(image_batch), np.array(label_batch)


images = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="tomImages")
labels = tf.placeholder(tf.float32, [batch_size, 10], name="tomLabels")
net = MyNet({'data': images})

ip2 = net.layers['ip2']
pred = tf.nn.softmax(ip2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=ip2), 0)
opt = tf.train.RMSPropOptimizer(0.001)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    # Load the data
    sess.run(tf.initialize_all_variables())
    net.load('mynet.npy', sess)

    data_gen = gen_data_batch(mnist.train)
    for i in range(1000):
        np_images, np_labels = next(data_gen)
        feed = {images: np_images, labels: np_labels}

        np_loss, np_pred, _ = sess.run([loss, pred, train_op], feed_dict=feed)
        if i % 10 == 0:
            print('Iteration: ', i, np_loss)

    print 'Done training!'




    # Yitao-TLS-Begin
    export_path_base = "caffe_mnist"
    export_path = os.path.join(
        compat.as_bytes(export_path_base),
        compat.as_bytes(str(FLAGS.model_version)))
    print 'Exporting trained model to', export_path
    builder = saved_model_builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(images)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(pred)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')
    # Yitao-TLS-End