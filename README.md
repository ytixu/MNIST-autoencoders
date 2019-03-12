# MNIST via pattern completion


### Autoencoding (flatten)
Test loss: 0.12039102125167847
Test accuracy: 0.05656530033349991

### Classification 
Input handwritten digit, output class as a probability vector

#### Feature extraction (flatten)
Test loss: 0.3349499772429466
Test accuracy: 0.9001

#### Pattern matching (flatten)
Test loss: 1.7224794626235962
Test accuracy: 0.7823

#### Pattern completion  (flatten)
Test loss: 1.9187672679901122
Test accuracy: 0.9314

#### Pattern completion (all) (flatten)
Test loss: 1.9410147674560547
Test accuracy: 0.9093

### Generation from labels 
Input one-hot encoded label, output handwritten digit.

#### End-to-End (flatten)
![Digit generation using end-to-end model](./images/flatten_generation_E2E.png)

#### Pattern completion (flatten)
![Digit generation using pattern completion](./images/flatten_generation_PCL.png)

#### Pattern completion (cnn)
![Digit generation using pattern completion](./images/cnn_generation_PCL.png)





