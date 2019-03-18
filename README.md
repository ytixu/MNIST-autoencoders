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
    <td>Pattern completion (FN)</td>
    <td>0.9353</td>
  </tr>
  <tr>
    <td>Pattern completion (FN, all)</td>
    <td>0.9093</td>
  </tr>
  <tr>
    <td rowspan="3">CNN</td>
    <td>Feature extraction</td>
    <td>0.8945</td>
  </tr>
  <!-- <tr>
    <td>Pattern matching</td>
    <td>0.7823</td>
  </tr> -->
  <tr>
    <td>Pattern completion (FN)</td>
    <td>0.9467</td>
  </tr>
  <tr>
    <td>Pattern completion (ADD)</td>
    <td>0.9491</td>
  </tr>
 <!--  <tr>
    <td>Pattern completion (all)</td>
    <td>0.9093</td>
  </tr> -->
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
    <td><img src="./images/flatten_generation_E2E.png" alt="Digit generation using end-to-end model"></td>
  </tr>
  <tr>
    <td>Pattern completion</td>
    <td><img src="./images/flatten_generation_PCL.png" alt="Digit generation using end-to-end model"></td>
  </tr>
  <tr>
    <td rowspan="2">CNN</td>
    <td>End-to-End</td>
    <td><img src="./images/cnn_generation_E2E.png" alt="Digit generation using end-to-end model"></td>
  </tr>
  <tr>
    <td>Pattern completion</td>
    <td><img src="./images/cnn_generation_PCL.png" alt="Digit generation using end-to-end model"></td>
  </tr>
</table>