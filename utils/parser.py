import argparse
import time

def get_timestamp_filename():
	return str(int(time.time()))+'.hdf5'

def parse():
	out_path = './out/'
	model_list = ['flatten', 'cnn']

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--random', action='store_true', help='Random uniform as zero value.')
	parser.add_argument('-1', '--ones', action='store_true', help='Use 1 as zero value.')
	parser.add_argument('--model', default='flatten', choices=model_list, type=str.lower, help='Autoencoder model.')
	parser.add_argument('-P', '--load_path', required=False, help='Model path.')
	args = vars(parser.parse_args())
 
 	if args['model'] == 'cnn':
 		args['save_path'] = out_path+'cnn/'+get_timestamp_filename()
 	else:
 		args['save_path'] = out_path+'flatten/'+get_timestamp_filename()

 	return args