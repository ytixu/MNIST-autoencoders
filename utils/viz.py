import math
import numpy as np
import matplotlib.pyplot as plt

#####
# Plot 10 digits in a row.
#####
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

#####
# Plot in concentric squares
#####
def plot_number(gen):
	n = int(math.sqrt(len(gen)))
	f, ax = plt.subplots(n, n, figsize=(n,n),
						gridspec_kw={'wspace':0.03, 'hspace':0.03}, 
						squeeze=True)

	mid_n = int(n/2)
	ax[mid_n, mid_n].axis("off")
	ax[mid_n, mid_n].imshow(gen[0], cmap='gray')

	image_index = 1
	for r in range(mid_n):
		i = r+1
		for c in range(mid_n-r-1, mid_n+r+1):
			ax[mid_n-i,c].axis("off")
			ax[mid_n-i,c].imshow(gen[image_index], cmap='gray')
			image_index += 1
			ax[c,mid_n+i].axis("off")
			ax[c,mid_n+i].imshow(gen[image_index], cmap='gray')
			image_index += 1
		for c in range(mid_n+r+1, mid_n-r-1, -1):
			ax[mid_n+i,c].axis("off")
			ax[mid_n+i,c].imshow(gen[image_index], cmap='gray')
			image_index += 1
			ax[c,mid_n-i].axis("off")
			ax[c,mid_n-i].imshow(gen[image_index], cmap='gray')
			image_index += 1
	plt.show()
	plt.close()


# def func(gen):
# 	n = int(math.sqrt(len(gen)))
# 	mid_n = int(n/2)
# 	res = np.zeros((n,n))
# 	print res
# 	res[mid_n,mid_n] = gen[0]
# 	image_index = 1
# 	for r in range(mid_n):
# 		i = r+1
# 		for c in range(mid_n-r-1, mid_n+r+1):
# 			res[mid_n-i,c] = gen[image_index]
# 			print res
# 			image_index += 1
# 			res[c,mid_n+i] = gen[image_index]
# 			image_index += 1
# 			print res
# 		for c in range(mid_n+r+1, mid_n-r-1, -1):
# 			res[mid_n+i,c] = gen[image_index]
# 			image_index += 1
# 			print res
# 			res[c,mid_n-i] = gen[image_index]
# 			image_index += 1
# 			print res

# func(a)


#####
# Plot 10x10 digits
#####
def plot_transition(gen):
	num_rows = 10
	num_cols = 10
	f, ax = plt.subplots(num_rows, num_cols, figsize=(num_rows,num_cols),
						gridspec_kw={'wspace':0.03, 'hspace':0.03}, 
						squeeze=True)
	for r in range(num_rows):
		for c in range(num_cols):
			image_index = r * num_cols + c
			ax[r,c].axis("off")
			ax[r,c].imshow(gen[image_index], cmap='gray')
			# ax[r,c].set_title('No. %d' % image_index)
	plt.show()
	plt.close()