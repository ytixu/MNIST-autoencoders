# MNIST via pattern completion


### Autoencoding

Model | Loss (`binary_cross_entropy`) | MAE
--- | --- | ----
Flatten | 0.1108 | 0.0493


### Classification 
Input handwritten digit, output class as a probability vector

Model | Method | Accuracy
--- | --- | ----
Flatten | 	Feature extraction<br> 
			Pattern matching<br> 
			Pattern completion<br>
			Pattern completion (all)| 	0.9077<br>
										0.7823<br>
										0.9353<br>
										0.9093



### Generation from labels 
Input one-hot encoded label, output handwritten digit.

Model | Method | Result
--- | --- | ----
Flatten | End-to-End<br>Pattern completion | ![Digit generation using end-to-end model](./images/flatten_generation_E2E.png)<br>![Digit generation using pattern completion](./images/flatten_generation_PCL.png)
CNN | End-to-End<br>Pattern completion | ![Digit generation using end-to-end model](./images/cnn_generation_E2E.png)<br>![Digit generation using pattern completion](./images/cnn_generation_PCL.png)





