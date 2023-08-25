## Implementing various Multimodal models (Visual and textual information) for Facebook Hateful memes competition. 

### Model 1:
A simple model that tries to learn textual features using attention model and learn Image features using pretrained models and then concatenate the result. However, the model has very less accuracy as it failed to see the relationship between textual and visual representation.

```
Model(
  (preLSTM): PreLSTM(
    (embed): Embedding(9137, 64)
    (biLSTM): LSTM(64, 64, bidirectional=True)
  )
  (postLSTM): PostLSTM(
    (lstm): LSTMCell(128, 128)
    (attn): Attn(
      (attn): Linear(in_features=256, out_features=1, bias=True)
    )
  )
  (concatenationModel): Concatenate(
    (final): Linear(in_features=128, out_features=1, bias=True)
    (image_layer): Linear(in_features=25088, out_features=1024, bias=True)
    (image_layer1): Linear(in_features=1024, out_features=64, bias=True)
    (final_one): Linear(in_features=456, out_features=64, bias=True)
    (drop): Dropout(p=0.2, inplace=False)
    (choose): Linear(in_features=64, out_features=1, bias=True)
    (biLSTM): LSTM(1, 2, num_layers=2, dropout=0.01, bidirectional=True)
  )
)
```
Result: Model Overfits, but underfits when number of Dropout Layers is increased.
```
Epoch: 82
[******************************]9/9   Loss:  0.00001
Training Accuracy:
Accuracy: 0.9977647058823529 || Loss: 0.00116
Validation Accuracy:
Accuracy: 0.502 || Loss: 0.30988
```

### Model 2:
Uses Glove Embedding to extract features from textual data and uses Convolution on features extracted from pretrained model to extract information about visual data. Then uses a concatenation model based on [this](https://arxiv.org/pdf/1908.04107.pdf) paper. Used K-fold validation to reduce overfitting.
```
Model(
  (text_model): textModel(
    (embed): Embedding(8882, 100)
    (mlp): Linear(in_features=100, out_features=64, bias=True)
    (dropout): Dropout(p=0.4, inplace=False)
  )
  (image_model): imageModel(
    (conv1): Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (fcx): Linear(in_features=128, out_features=64, bias=True)
  (fcy): Linear(in_features=49, out_features=64, bias=True)
  (batchnorm1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (batchnorm2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (UA_model1): UA(
    (gsaunit): GSA(
      (fcv): Linear(in_features=64, out_features=64, bias=True)
      (fck): Linear(in_features=64, out_features=64, bias=True)
      (fcq): Linear(in_features=64, out_features=64, bias=True)
      (gdp): GatedDotProduct(
        (fcg): Linear(in_features=64, out_features=2, bias=True)
      )
    )
    (batchnorm1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (dropout): Dropout(p=0.4, inplace=False)
  )
  (categorize): Categorize(
    (fc): Linear(in_features=64, out_features=1, bias=True)
    (conv2): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (conv3): Conv1d(64, 32, kernel_size=(5,), stride=(1,), padding=(2,))
    (maxpooling): MaxPool1d(kernel_size=64, stride=64, padding=0, dilation=1, ceil_mode=False)
    (dropout): Dropout(p=0.3, inplace=False)
  )
)
```
Result: Again Overfitting. 
```
Training Accuracy:
Accuracy: 0.9945882352941177 || Loss: 0.00003
Validation Accuracy:
Accuracy: 0.534 || Loss: 0.01501
```

### Model 3:
Uses text filtering and then GLoVE Word Embeddings for textual feature extraction and trained model for visual feature extraction. Then uses same concatenation model as above for output.
```
Model(
  (text_model): textModel(
    (embed): Embedding(8882, 100)
    (mlp): Linear(in_features=100, out_features=64, bias=True)
    (dropout): Dropout(p=0.4, inplace=False)
    (LSTM): LSTM(64, 64, batch_first=True, bidirectional=True)
  )
  (fcx): Linear(in_features=128, out_features=128, bias=True)
  (fcy): Linear(in_features=49, out_features=128, bias=True)
  (batchnorm1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (batchnorm2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (UA_model1): UA(
    (gsaunit): GSA(
      (fcv): Linear(in_features=128, out_features=128, bias=True)
      (fck): Linear(in_features=128, out_features=128, bias=True)
      (fcq): Linear(in_features=128, out_features=128, bias=True)
    )
    (batchnorm1): BatchNorm1d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (UA_model2): UA(
    (gsaunit): GSA(
      (fcv): Linear(in_features=128, out_features=128, bias=True)
      (fck): Linear(in_features=128, out_features=128, bias=True)
      (fcq): Linear(in_features=128, out_features=128, bias=True)
    )
    (batchnorm1): BatchNorm1d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (categorize): Categorize(
    (fc1): Linear(in_features=69632, out_features=512, bias=True)
    (fc2): Linear(in_features=512, out_features=256, bias=True)
    (fc3): Linear(in_features=256, out_features=64, bias=True)
    (fc4): Linear(in_features=64, out_features=1, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
)
```
Result: Overfitting with some improvement.
```
Training Accuracy:
Accuracy: 0.971764705882353 || Loss: 0.00013
Validation Accuracy:
Accuracy: 0.524 || Loss: 0.00633
```
### Model 4:
Use Some text filtering and then used GLoVE word embeddings for feature extraction and trained model for visual feature extraction. Simplified the above concatenation model to reduce the complexity and thus reduce overfitting.
```
```
Result:
```
```

### Conclusion:
1. All these models possibly lose features during extraction. Can try a better feature extaction technique.
2. Can try data augmentation to reduce overfitting.

#### Thanks
