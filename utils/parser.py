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
	model_list = ['flatten', 'cnn']

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--random', action='store_true', help='Random uniform as zero value.')
	parser.add_argument('-1', '--ones', action='store_true', help='Use 1 as zero value.')
	parser.add_argument('--model', default='flatten', choices=model_list, type=str.lower, help='Autoencoder model.')
	parser.add_argument('-P', '--load_path', required=False, help='Model path.')
	args = vars(parser.parse_args())
 
 	if args['model'] == 'cnn':
 		args['save_path'] = get_dir('cnn')+get_timestamp_filename()
 	else:
 		args['save_path'] = get_dir('flatten')+get_timestamp_filename()

 	return args