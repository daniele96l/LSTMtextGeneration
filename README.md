# LSTMtextGeneration

In this work we implemented and trained a combination of a recurrent neural network
and a convolutional neural network to answer questions related to images. 

The dataset was quite big but still, we decided to keep all train data and use them 
at once, without splitting the training in different phases with subsets of the 
training set.

The training input to the model is a tuple made by an image and a question, and the prediction
corresponds to a class where each class is mapped to a specific answer to a possible question.

To keep low the usage of memory we decided to load images only when they need to be used, so each
image is loaded when the train has already started, when a train object tries to look for it for the
first time. Of course, this highly affects the training speed, which becomes quite low.

The first model we tried to use was combining the simple VGG16 and a recurrent network
made by an Encoder layer and an LSTM layer through a Concatenate layer to make the 
prediction on the class of answer. This model was already quite good so for the next models
we tried to change some parameters like the number of units in the LSTM layers and adding a 
further dense layer before the softmax. This slightly improved the performance of the model.
We also tried to add Dropout layers to avoid the relatively early overfitting on data, but also
this option didn't improve much the performance.

Furtherly, we tried to substitute the LSTM layer with GTU layer, but the difference between the
two was irrelevant so we just turned back to the LSTM layer.

The best model that we found uses the VGG19 as convolutional network followd by two dense layers,
a recurrent network made by an Embedding layer followed by two LSTM layers with differente number 
of units, then the two models are concatenated and fed into a dense layer followed by the classification
layer with the softmax. The accuracy of this model is slightly below 60%
