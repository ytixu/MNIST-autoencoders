# MNIST via Pattern Completion Learning

Pattern completion learning (PCL) is an inference strategy where given data pairs (X,Y), a PCL model tries to learn correlation between the latent representations of the partial pattern (X) and the complete pattern (XY).


### Autoencoding

Model | Loss (`binary_cross_entropy`) | L1 Distance
--- | --- | ----
Flatten | 0.1109 | 0.0492
CNN |  0.0897 | 0.0314



### Classification 
Input handwritten digit, output class as a probability vector

<table>
  <tr>
    <th rowspan="2">Learning strategy</th>
    <th rowspan="2">Latent mapping</th>
    <th colspan="2">Accuracy</th>
  </tr>
  <tr>
    <td>Flatten</td>
    <td>CNN</td>
  </tr>
  <tr>
    <td>Feature extraction</td>
    <td>-</td>
    <td>0.8989</td>
    <td>0.8362</td>
  </tr>
  <tr>
    <td rowspan="2">Pattern matching</td>
    <td>FN</td>
    <td>0.7733</td>
    <td>0.7035</td>
  </tr>
  <tr>
    <td>(ADD)</td>
    <td>0.7457</td>
    <td>0.5832</td>
  </tr>
  <tr>
    <td rowspan="2">Pattern completion</td>
    <td>FN</td>
    <td>0.9212</td>
    <td>0.9741</td>
  </tr>
  <tr>
    <td>(ADD)</td>
    <td>0.9240</td>
    <td>0.9723</td>
  </tr>
  <tr>
    <td rowspan="2">Pattern completion (all)</td>
    <td>FN</td>
    <td>0.9097</td>
    <td>0.9067</td>
  </tr>
  <tr>
    <td>(ADD)</td>
    <td>0.8498</td>
    <td>0.7641</td>
  </tr>
  <!-- <tr>
    <td rowspan="7">CNN</td>
    <td>Feature extraction</td>
    <td>0.8800</td>
  </tr>
  <tr>
    <td>Pattern matching (FN)</td>
    <td>0.5680</td>
  </tr>
  <tr>
    <td>Pattern matching (ADD)</td>
    <td>0.3478</td>
  </tr>
  <tr>
    <td>Pattern completion (FN)</td>
    <td>0.9666</td>
  </tr>
  <tr>
    <td>Pattern completion (ADD)</td>
    <td>0.9679</td>
  </tr>
  <tr>
    <td>Pattern completion (FN, all)</td>
    <td>0.8865</td>
  </tr>
  <tr>
    <td>Pattern completion (ADD, all)</td>
    <td>0.6913</td>
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
    <td rowspan="3">Flatten</td>
    <td>End-to-End</td>
    <td><img src="./images/flatten_generation_E2E.png" alt="Digit generation using end-to-end model"></td>
  </tr>
  <tr>
    <td>Pattern completion (ADD)</td>
    <td><img src="./images/flatten_generation_PCL-add.png" alt="Digit generation using PCL model"></td>
  </tr>
  <tr>
    <td>Pattern completion (FN)</td>
    <td><img src="./images/flatten_generation_PCL.png" alt="Digit generation using PCL model"></td>
  </tr>
  <tr>
    <td rowspan="3">CNN</td>
    <td>End-to-End</td>
    <td><img src="./images/cnn_generation_E2E.png" alt="Digit generation using end-to-end model"></td>
  </tr>
  <tr>
    <td>Pattern completion (ADD)</td>
    <td><img src="./images/cnn_generation_PCL-add.png" alt="Digit generation using PCL model"></td>
  </tr>
  <tr>
    <td>Pattern completion (FN)</td>
    <td><img src="./images/cnn_generation_PCL.png" alt="Digit generation using PCL model"></td>
  </tr>
</table>

Adding gaussian noise to the latent representation of the generated digit. The center digit has zero noise, the digits on the first layer around the center has 50% of the mean STD, and those on the last layer has 100% of the mean STD. 

<table>
  <tr>
    <td>Flatten</td>
    <td>CNN</td>
  </tr>
  <tr>
    <td><img width="400px" src="./images/flatten/flatten_neighbours.gif" alt="Digit generation using PCL model"></td>
    <td><img width="400px" src="./images/cnn/cnn_neighbours.gif" alt="Digit generation using PCL model"></td>
  </tr>
</table>
