# MNIST via pattern completion


### Autoencoding (flatten)
Test loss: 0.11078792009353637
Test accuracy: 0.049322482126951214

### Classification 
Input handwritten digit, output class as a probability vector

#### Feature extraction (flatten)
Test loss: 0.32502294434309004
Test accuracy: 0.9077

#### Pattern matching (flatten)
Test loss: 1.7224794626235962
Test accuracy: 0.7823

#### Pattern completion  (flatten)
Test loss: 1.9041902032852174
Test accuracy: 0.9353

#### Pattern completion (all) (flatten)
Test loss: 1.9410147674560547
Test accuracy: 0.9093

### Generation from labels 
Input one-hot encoded label, output handwritten digit.

#### End-to-End (flatten)
![Digit generation using end-to-end model](./images/flatten_generation_E2E.png)

#### End-to-End (CNN)
![Digit generation using end-to-end model](./images/cnn_generation_E2E.png)

#### Pattern completion (flatten)
![Digit generation using pattern completion](./images/flatten_generation_PCL.png)

#### Pattern completion (cnn)
![Digit generation using pattern completion](./images/cnn_generation_PCL.png)





