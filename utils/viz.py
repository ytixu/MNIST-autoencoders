import numpy as np
import matplotlib.pyplot as plt

def plot(gen):
	num_rows = 1
	num_cols = 10
	f, ax = plt.subplots(num_rows, num_cols, figsize=(10,1),
						gridspec_kw={'wspace':0.03, 'hspace':0.01}, 
						squeeze=True)
	for r in range(num_rows):
		for c in range(num_cols):
			image_index = r * num_cols + c
			ax[c].axis("off")
			# ax[r,c].axis("off")
			ax[c].imshow(gen[image_index], cmap='gray')
			# ax[r,c].imshow(gen[image_index], cmap='gray')
			# ax[r,c].set_title('No. %d' % image_index)
	plt.show()
	plt.close()