import argparse

def parse():
	model_list = ['flatten', 'cnn']
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--random', action='store_true', help='Random uniform as zero value.')
	parser.add_argument('-1', '--ones', action='store_true', help='Use 1 as zero value.')
	parser.add_argument('--model', default='flatten', choices=model_list, type=str.lower, help='Autoencoder model.')
	args = vars(parser.parse_args())
 
 	return args