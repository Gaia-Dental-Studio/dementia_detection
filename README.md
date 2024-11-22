# Dementia Detection

### Technical Information (Dementia Model v2)
This model was trained with 2D-CNN architecture on [The augmented Alzheimer MRI datasets](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data) using Keras. The model is available [here](https://drive.google.com/drive/folders/1gTFjKmHUug2FAzspefsW3FvJHmqvte3-?usp=drive_link). The training was run with batch size 32, learning rate 0.001, and SGD optimizer. However, the iteration number (epoch) is set differently for CPU and GPU. The accuracy of both training processes is as follows:

<table class="table table-bordered">
  <thead class="thead-light">
    <tr>
      <th>Parameter</th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Epoch</strong></td>
      <td>10</td>
      <td>30</td>
    </tr>
    <tr>
      <td><strong>Training Accuracy</strong></td>
      <td>88%</td>
      <td>99%</td>
    </tr>
    <tr>
      <td><strong>Evaluation Accuracy</strong></td>
      <td>85%</td>
      <td>91%</td>
    </tr>
  </tbody>
</table>


### API Endpoint
Dementia model v2: multiclass-api_v2.py

Dementia model: multiclass-api.py
