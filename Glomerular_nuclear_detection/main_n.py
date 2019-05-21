import argparse
import os
import tensorflow as tf
from model import Model



"""
This script defines hyperparameters. It has been slightly modified to accept an input prior weighting value, to bias the network towards segmentation of class 1 (nuclei).
"""



def configure():
	flags = tf.app.flags

	# training
	flags.DEFINE_integer('num_steps', 300000, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', 20000, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 0.0025, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model, res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', './modelspretrain/deeplab_resnet.ckpt', 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', './dataset/JASN_train_split.txt', 'training data list filename')

	# validation
	flags.DEFINE_integer('valid_step', 300000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 19, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', './dataset/val.txt', 'validation data list filename')

	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', 'output_JASN_holdout_2', 'directory for saving outputs')
	flags.DEFINE_integer('test_step', 300000, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', 1, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list', './dataset/test.txt', 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')
	flags.DEFINE_float('prior', 0.1, 'Amount to weight predictions towards the nuclear class (0.5 is balanced)')

	# data
	flags.DEFINE_string('data_dir', '/hdd/BG_projects/JASN/NuclearSegmentation_test/', 'data directory')
	flags.DEFINE_integer('batch_size', 20, 'training batch size')
	flags.DEFINE_integer('input_height', 128, 'input image height')
	flags.DEFINE_integer('input_width', 128, 'input image width')
	flags.DEFINE_integer('num_classes', 2, 'number of classes')
	flags.DEFINE_integer('ignore_label', 5, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', False, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')

	# log
	flags.DEFINE_string('modeldir', 'model_JASN_review_full', 'model directory')
	flags.DEFINE_string('logfile', 'log_split.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log_split', 'training log directory')

	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='train',
		help='actions: train, test, or predict')

	args = parser.parse_args()

	if args.option not in ['train', 'test', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")
	else:
		# Set up tf session and initialize variables.
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)
		sess = tf.Session()
		# Run

		model = Model(sess, configure())
		getattr(model, args.option)()


if __name__ == '__main__':
	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	tf.app.run()
