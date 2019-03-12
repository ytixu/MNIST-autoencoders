# MNIST via pattern completion


### Autoencoding

Model | Loss (`binary_cross_entropy`) | MAE
--- | --- | ----
Flatten | 0.1108 | 0.0493


### Classification 
Input handwritten digit, output class as a probability vector

<table>
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td rowspan="4">Flatten</td>
    <td>Feature extraction</td>
    <td>0.9077</td>
  </tr>
  <tr>
    <td>Pattern matching</td>
    <td>0.7823</td>
  </tr>
  <tr>
    <td>Pattern completion</td>
    <td>0.9353</td>
  </tr>
  <tr>
    <td>Pattern completion (all)</td>
    <td>0.9093</td>
  </tr>
</table>



### Generation from labels 
Input one-hot encoded label, output handwritten digit.

<table>
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th>Result</th>
  </tr>
  <tr>
    <td rowspan="2">Flatten</td>
    <td>End-to-End</td>
    <td>![Digit generation using end-to-end model](./images/flatten_generation_E2E.png)</td>
  </tr>
  <tr>
    <td>Pattern completion</td>
    <td>![Digit generation using pattern completion](./images/flatten_generation_PCL.png)</td>
  </tr>
  <tr>
    <td rowspan="2">Flatten</td>
    <td>End-to-End</td>
    <td>![Digit generation using end-to-end model](./images/cnn_generation_E2E.png)</td>
  </tr>
  <tr>
    <td>Pattern completion</td>
    <td>![Digit generation using pattern completion](./images/cnn_generation_PCL.png)</td>
  </tr>
</table>