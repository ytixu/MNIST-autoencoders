import os
import argparse
import time

def get_dir(folder):
	directory = './out/'+folder
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory+'/'

def get_timestamp_filename():
	return str(int(time.time()))+'.hdf5'

def parse():
	model_list = ['flatten', 'dense_cnn', 'cnn', 'rw_cnn']

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--random', action='store_true', help='Random uniform as zero value.')
	parser.add_argument('-1', '--ones', action='store_true', help='Use 1 as zero value.')
	parser.add_argument('-n', '--neg', action='store_true', help='Use -1 as zero value.')
	parser.add_argument('--model', default='flatten', choices=model_list, type=str.lower, help='Autoencoder model.')
	parser.add_argument('-P', '--load_path', required=False, help='Model path.')
	parser.add_argument('-d', '--no-display', action='store_true', help='Plot graphs.')
	parser.add_argument('-c', '--extra-class', action='store_true', help='Add extra class for no label.')
	args = vars(parser.parse_args())

	#if args['model'] == 'cnn':
	#	args['save_path'] = get_dir('cnn')+get_timestamp_filename()
	#elif args['model'] == 'dense_cnn':
	#	args['save_path'] = get_dir('dense_cnn')+get_timestamp_filename()
	#else:
	args['save_path'] = get_dir(args['model'])+get_timestamp_filename()

	if 'no_display' not in args:
		args['no_display'] = False
	if 'extra_class' not in args:
		args['extra_class'] = False

	return args
