#coding:utf-8
import tensorflow as tf
FLAGS = tf.app.flags
#flag设置
tf.app.flags.DEFINE_string(
    'dataset_name', 'coco',
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'coco',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', './',
    'Output directory where to store TFRecords files.')
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('the datasetname must be defined')
    else:
        dataset_path = FLAGS.dataset_dir



