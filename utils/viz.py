import math
import numpy as np
import matplotlib.pyplot as plt

#####
# Plot 10 digits in a row.
#####
def plot(imgs):
	num_rows = 1
	num_cols = 10
	f, ax = plt.subplots(num_rows, num_cols, figsize=(10,1),
						gridspec_kw={'wspace':0.03, 'hspace':0.01}, 
						squeeze=True)
	for r in range(num_rows):
		for c in range(num_cols):
			image_index = r * num_cols + c
			ax[c].axis("off")
			ax[c].imshow(imgs[image_index], cmap='gray')
	plt.show()
	plt.close()

#####
# Plot in concentric squares
#####
def plot_number(imgs):
	n = int(math.sqrt(len(imgs)))
	f, ax = plt.subplots(n, n, figsize=(n,n),
						gridspec_kw={'wspace':0.03, 'hspace':0.03}, 
						squeeze=True)

	mid_n = int(n/2)
	ax[mid_n, mid_n].axis("off")
	ax[mid_n, mid_n].imshow(imgs[0], cmap='gray')

	image_index = 1
	for r in range(mid_n):
		i = r+1
		for c in range(mid_n-r-1, mid_n+r+1):
			ax[mid_n-i,c].axis("off")
			ax[mid_n-i,c].imshow(imgs[image_index], cmap='gray')
			image_index += 1
			ax[c,mid_n+i].axis("off")
			ax[c,mid_n+i].imshow(imgs[image_index], cmap='gray')
			image_index += 1
		for c in range(mid_n+r+1, mid_n-r-1, -1):
			ax[mid_n+i,c].axis("off")
			ax[mid_n+i,c].imshow(imgs[image_index], cmap='gray')
			image_index += 1
			ax[c,mid_n-i].axis("off")
			ax[c,mid_n-i].imshow(imgs[image_index], cmap='gray')
			image_index += 1
	plt.show()
	plt.close()



#####
# Plot 10x10 digits
#####
def plot_matrix(imgs, title=None):
	num_rows = 10
	num_cols = 10
	f, ax = plt.subplots(num_rows, num_cols, figsize=(num_rows,num_cols),
						gridspec_kw={'wspace':0.03, 'hspace':0.03}, 
						squeeze=True)
	for r in range(num_rows):
		for c in range(num_cols):
			image_index = r * num_cols + c
			plt.setp(ax[r,c].get_xticklabels(), visible=False)
			plt.setp(ax[r,c].get_yticklabels(), visible=False)
			ax[r,c].tick_params(axis=u'both', which=u'both',length=0)
			ax[r,c].imshow(imgs[image_index], cmap='gray')
			# ax[r,c].set_title('No. %d' % image_index)

	for c in range(num_cols):
		ax[0,c].set_title(c)
	# for r in range(num_rows):
		ax[c,0].set_ylabel(c, rotation='horizontal', labelpad=20)

	if title:
		f.suptitle(title)

	plt.show()
	plt.close()