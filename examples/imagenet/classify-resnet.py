#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

import models
import dataset

# Yitao-TLS-Begin
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End

def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('imagenet-classes.txt', 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))


def classify(model_data_path, image_paths):
    '''Classify the given images using GoogleNet.'''

    # Get the data specifications for the GoogleNet model
    spec = models.get_data_spec(model_class=models.GoogleNet)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network
    net = models.GoogleNet({'data': input_node})

    # Create an image producer (loads and processes images in parallel)
    image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)

    with tf.Session() as sesh:
        # Start the image processing workers
        coordinator = tf.train.Coordinator()
        threads = image_producer.start(session=sesh, coordinator=coordinator)

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sesh)

        # Load the input image
        print('Loading the images')
        indices, input_images = image_producer.get(sesh)

        # Perform a forward pass through the network to get the class probabilities
        print('Classifying')
        probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})

        display_results([image_paths[i] for i in indices], probs)

        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)

def check_is_jpeg(path):
    extension = osp.splitext(path)[-1].lower()
    if extension in ('.jpg', '.jpeg'):
        return True
    if extension != '.png':
        raise ValueError('Unsupported image format: {}'.format(extension))
    return False

def process_image(img, scale, isotropic, crop, mean):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.stack([scale, scale])
    img = tf.image.resize_images(img, new_shape)
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.stack([offset[0], offset[1], 0]), size=tf.stack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean

def load_image(image_path, is_jpeg, data_spec):
    # Read the file
    file_data = tf.read_file(image_path)
    # Decode the image data
    img = tf.image.decode_jpeg(file_data, channels=data_spec.channels)
    # img = tf.cond(
    #     is_jpeg,
    #     lambda: tf.image.decode_jpeg(file_data, channels=data_spec.channels),
    #     lambda: tf.image.decode_png(file_data, channels=data_spec.channels))
    if data_spec.expects_bgr:
        # Convert from RGB channel ordering to BGR
        # This matches, for instance, how OpenCV orders the channels.
        # img = tf.reverse(img, [False, False, True])
        img = tf.reverse(img,[2])
    return img

def tomClassify(model_data_path, image_paths):
    '''Classify the given images using GoogleNet.'''

    # Get the data specifications for the GoogleNet model
    spec = models.get_data_spec(model_class=models.GoogleNet)

    # Create a placeholder for the input image
    # input_node = tf.placeholder(tf.float32,
    #                             shape=(None, spec.crop_size, spec.crop_size, spec.channels))
    input_img_name = tf.placeholder(tf.string)
    image_path = image_paths[0]
    is_jpeg = check_is_jpeg(image_path)
    img = load_image(input_img_name, is_jpeg, spec)
    processed_img = process_image(img=img,
                                      scale=spec.scale_size,
                                      isotropic=spec.isotropic,
                                      crop=spec.crop_size,
                                      mean=spec.mean)

    input_node = tf.reshape(processed_img, [1, 224, 224, 3])
    
    # Construct the network
    net = models.GoogleNet({'data': input_node})

    pred = net.layers['prob']

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        net.load(model_data_path, sess)

        probs = sess.run(net.get_output(), feed_dict={input_img_name: image_path})

        print(probs.shape)

        display_results([image_path], probs)

        print("Done classify one image!")

        # Yitao-TLS-Begin
        export_path_base = "caffe_googlenet"
        export_path = os.path.join(
            compat.as_bytes(export_path_base),
            compat.as_bytes(str(FLAGS.model_version)))
        print 'Exporting trained model to', export_path
        builder = saved_model_builder.SavedModelBuilder(export_path)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_img_name)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(net.get_output())

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'image_name': tensor_info_x},
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

def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  image = tf.image.central_crop(image, central_fraction=0.875)
  # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(
      image, [224, 224], align_corners=False)
  image = tf.squeeze(image, [0])
  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image

def tomClassify2(model_data_path):
    '''Classify the given images using GoogleNet.'''

    # Get the data specifications for the GoogleNet model
    # spec = models.get_data_spec(model_class=models.ResNet50)
    spec = models.get_data_spec(model_class=models.ResNet152)

    # Create a placeholder for the input image
    # input_node = tf.placeholder(tf.float32,
                                # shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network

    # Create an image producer (loads and processes images in parallel)
    # image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)

    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(
            shape=[], dtype=tf.string),
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    jpegs = tf_example['image/encoded']
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

    # net = models.ResNet50({'data': images})    
    net = models.ResNet152({'data': images})    

    with tf.Session() as sess:
        # Start the image processing workers
        # coordinator = tf.train.Coordinator()
        # threads = image_producer.start(session=sess, coordinator=coordinator)

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)

        # Load the input image
        # print('Loading the images')
        # indices, input_images = image_producer.get(sess)

        # Perform a forward pass through the network to get the class probabilities
        # print('Classifying')
        # probs = sess.run(net.get_output(), feed_dict={input_node: input_images})
        # display_results([image_paths[i] for i in indices], probs)

        # # Stop the worker threads
        # coordinator.request_stop()
        # coordinator.join(threads, stop_grace_period_secs=2)

        # Yitao-TLS-Begin
        # export_path_base = "caffe_resnet50"
        export_path_base = "caffe_resnet152"
        export_path = os.path.join(
            compat.as_bytes(export_path_base),
            compat.as_bytes(str(FLAGS.model_version)))
        print 'Exporting trained model to', export_path
        builder = saved_model_builder.SavedModelBuilder(export_path)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(jpegs)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(net.get_output())

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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the GoogleNet model')
    parser.add_argument('image_paths', nargs='+', help='One or more images to classify')
    args = parser.parse_args()

    # Classify the image
    # classify(args.model_path, args.image_paths)
    # tomClassify(args.model_path, args.image_paths)
    tomClassify2(args.model_path)


if __name__ == '__main__':
    main()
