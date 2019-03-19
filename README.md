# MNIST via Pattern Completion Learning

Pattern completion learning (PCL) is an inference strategy where given data pairs (X,Y), a PCL model tries to learn correlation between the latent representations of the partial pattern (X) and the complete pattern (XY).


### Autoencoding

Model | Loss (`binary_cross_entropy`) | L1 Distance
--- | --- | ----
Flatten | 0.1340 | 0.0683
CNN |  0.1019 | 0.0399



### Classification 
Input handwritten digit, output class as a probability vector

<table>
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td rowspan="7">Flatten</td>
    <td>Feature extraction</td>
    <td>0.9077</td>
  </tr>
  <tr>
    <td>Pattern matching (FN)</td>
    <td>0.8304</td>
  </tr>
  <tr>
    <td>Pattern matching (ADD)</td>
    <td>0.5569</td>
  </tr>
  <tr>
    <td>Pattern completion (FN)</td>
    <td>0.9120</td>
  </tr>
  <tr>
    <td>Pattern completion (ADD)</td>
    <td>0.9126</td>
  </tr>
  <tr>
    <td>Pattern completion (FN, all)</td>
    <td>0.8948</td>
  </tr>
  <tr>
    <td>Pattern completion (ADD, all)</td>
    <td>0.8152</td>
  </tr>
  <tr>
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
    <td><img src="./images/flatten_generation_E2E.png" alt="Digit generation using end-to-end model"></td>
  </tr>
  <tr>
    <td>Pattern completion (FN)</td>
    <td><img src="./images/flatten_generation_PCL.png" alt="Digit generation using PCL model"></td>
  </tr>
  <tr>
    <td rowspan="2">CNN</td>
    <td>End-to-End</td>
    <td><img src="./images/cnn_generation_E2E.png" alt="Digit generation using end-to-end model"></td>
  </tr>
  <tr>
    <td>Pattern completion (FN)</td>
    <td><img src="./images/cnn_generation_PCL.png" alt="Digit generation using PCL model"></td>
  </tr>
</table>

<table>
  <tr>
    <td>Flatten</td>
    <td>CNN</td>
  </tr>
  <tr>
    <td><img src="./images/flatten/flatten_neighbours.gif" alt="Digit generation using PCL model"></td>
    <td><img src="./images/cnn/cnn_neighbours.gif" alt="Digit generation using PCL model"></td>
  </tr>
</table>
