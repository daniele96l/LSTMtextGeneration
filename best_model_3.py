#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import random
import math
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import metrics


# In[2]:


bs = 32

SEED = 6669

img_h = 160  # maintain aspect ratio of original images -> currently is / 2.5
img_w = 280

dataset_dir = "./VQA_Dataset"

assert(os.path.exists(dataset_dir))

images_dir = os.path.join(dataset_dir, "Images")

assert(os.path.exists(images_dir))


# In[3]:


num_classes = 58

labels_dict = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        'apple': 6,
        'baseball': 7,
        'bench': 8,
        'bike': 9,
        'bird': 10,
        'black': 11,
        'blanket': 12,
        'blue': 13,
        'bone': 14,
        'book': 15,
        'boy': 16,
        'brown': 17,
        'cat': 18,
        'chair': 19,
        'couch': 20,
        'dog': 21,
        'floor': 22,
        'food': 23,
        'football': 24,
        'girl': 25,
        'grass': 26,
        'gray': 27,
        'green': 28,
        'left': 29,
        'log': 30,
        'man': 31,
        'monkey bars': 32,
        'no': 33,
        'nothing': 34,
        'orange': 35,
        'pie': 36,
        'plant': 37,
        'playing': 38,
        'red': 39,
        'right': 40,
        'rug': 41,
        'sandbox': 42,
        'sitting': 43,
        'sleeping': 44,
        'soccer': 45,
        'squirrel': 46,
        'standing': 47,
        'stool': 48,
        'sunny': 49,
        'table': 50,
        'tree': 51,
        'watermelon': 52,
        'white': 53,
        'wine': 54,
        'woman': 55,
        'yellow': 56,
        'yes': 57
}


# In[4]:


class Train_Data():

  def __init__(self, id, object):
    print("Creating object ", id )
    self.object_id = id
    self.question = object['question']
    self.image_id = object['image_id']

    ans = labels_dict[object['answer']]
    self.answer = ans
    #assert(os.path.exists(os.path.join(images_dir, self.image_id + ".png")))

    #if id == "117792":
    # display(self.image)
      

  def print_object(self):
    print("start object " + str(self.object_id))
    print("Question: " + str(self.question))
    print("Answer: " + str(self.answer))



class Test_Data():

  def __init__(self, id, object):
    self.object_id = id
    self.question = object['question']
    self.image_id = object['image_id']
    image_ = image.load_img(os.path.join(images_dir, self.image_id + ".png"), target_size=(img_h, img_w))
    image_ = image.img_to_array(image_)
    
    self.image = image_

  def print_object(self):
    print("start object " + str(self.object_id))
    print("Question: " + str(self.question))


# In[5]:


class Dataset(tf.keras.utils.Sequence):

  def __init__(self, train_data):

    print("Creating a dataset")
    self.img_data_gen = ImageDataGenerator()
    self.train_data = train_data

  def __len__(self):
    #print("x")
    return len(self.train_data)

  def __getitem__(self, index):

    #print("y")

    data_item = self.train_data[index]

    if not(hasattr(data_item, 'image')):
      image_ = image.load_img(os.path.join(images_dir, data_item.image_id + ".png"), target_size=(img_h, img_w))
      image_ = image.img_to_array(image_)
      #image = Image.open(os.path.join(images_dir, data_item.image_id + ".png"))
      #image = image.resize((img_w, img_h), Image.ANTIALIAS)

      #data_item.image = np.array(image)
      data_item.image = image_
      

    return (data_item.image, data_item.question), data_item.answer 


# In[6]:


models_dir = os.path.join(dataset_dir, "models")
if (not os.path.exists(models_dir)):
  os.makedirs(models_dir)

prediction_dir = os.path.join(dataset_dir, "predictions")
if (not os.path.exists(prediction_dir)):
  os.makedirs(prediction_dir)

ckpt_dir = os.path.join(dataset_dir, "ckpts")
if (not os.path.exists(ckpt_dir)):
  os.makedirs(ckpt_dir)


# In[7]:


with open(os.path.join(dataset_dir, 'train_questions_annotations.json')) as f:    #read the file
  train_questions = json.load(f)

dataset = []
questions = []
answers = []

count = 0
for key, value in train_questions.items():
  train_item = Train_Data(key, value)
  dataset.append(train_item)

  questions.append(train_item.question)
  answers.append(train_item.answer)

  count += 1
  print("Created " + str(count) + " train items")

#for train in train_set:      #integrity check of the train objects
  #train.print_object()

#print("Number of questions: ", len(train_set))


# In[8]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create Tokenizer to convert words to integers
question_tokenizer = Tokenizer(num_words=10000)  #TODO: set parameter num_words
question_tokenizer.fit_on_texts(questions)
question_tokenized = question_tokenizer.texts_to_sequences(questions)

question_wtoi = question_tokenizer.word_index
print('Total question words:', len(question_wtoi))

max_question_length = max(len(sentence) for sentence in question_tokenized)
print('Max question sentence length:', max_question_length)


# In[9]:


question_encoder_inputs = pad_sequences(question_tokenized, maxlen=max_question_length)

print("Question encoder inputs shape:", question_encoder_inputs.shape)


# In[10]:


validation_split = 0.8

#train_file_im = os.path.join(train_imgs_dir, "Train_images.npy")

#train_images = np.load(train_file_im)

#assert(train_images)
#print("Image arrays file loaded")

for idx in range(len(dataset)):
  dataset[idx].question = question_encoder_inputs[idx]
  #dataset[idx].image = train_images[dataset[idx].image_id]

  #print(dataset[idx].question, dataset[idx].answer)

random.Random(SEED).shuffle(dataset)

bound = math.floor(len(dataset)*validation_split)
train_ = np.array(dataset[:bound])
val_ = np.array(dataset[bound:])

train_data = Dataset(train_)
val_data = Dataset(val_)

train_dataset = tf.data.Dataset.from_generator( lambda: train_data,
                                                 output_types=((tf.float32, tf.float32), tf.float32),
                                                output_shapes=(([img_h, img_w, None], [max_question_length]), ()))
  
val_dataset = tf.data.Dataset.from_generator( lambda: val_data,
                                                 output_types=((tf.float32, tf.float32), tf.float32),
                                                output_shapes=(([img_h, img_w, None], [max_question_length]), ()))

train_dataset = train_dataset.batch(bs)
train_dataset = train_dataset.repeat()

val_dataset = val_dataset.batch(bs)
val_dataset = val_dataset.repeat()


# In[11]:


# Optimization params
# -------------------

# Loss
# Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
loss = tf.keras.losses.SparseCategoricalCrossentropy() 

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------
metrics = ['accuracy']
# ------------------


# In[12]:


from datetime import datetime

now = datetime.now().strftime('%b%d_%H-%M-%S')

exp_name = 'visual_question_answering'

exp_dir = os.path.join(dataset_dir, exp_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    
callbacks = []

# Model checkpoint
# ----------------

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

# ----------------

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join(dataset_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
    
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

# Early Stopping
# --------------
early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callbacks.append(es_callback)

# ---------------------------------


# In[13]:


def get_model():

  ##### Create Convolutional Network

  model_img = tf.keras.Sequential()

  vgg = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=[img_h, img_w, 3])
  
  non_trainable_layers = 15
  for layer in range(non_trainable_layers):
    vgg.layers[layer].trainable = False

  model_img.add(vgg)

  model_img.add(Flatten())
  model_img.add(tf.keras.layers.Dense(units=1024, activation="relu"))
  model_img.add(tf.keras.layers.Dense(units=256, activation="relu"))

  image_input = Input(shape=[img_h, img_w, 3])
  encoded_image = model_img(image_input)

  #### Create Recurrent Network

  EMBEDDING_SIZE = 32

  question_input = tf.keras.Input(shape=[max_question_length])
  question_embedding_layer = tf.keras.layers.Embedding(len(question_wtoi)+1, EMBEDDING_SIZE, input_length=max_question_length, mask_zero=True)
  encoder_embedding_out = question_embedding_layer(question_input)
  encoder_quest = tf.keras.layers.LSTM(units=128, return_sequences=True)
  encoder_quest_2 = tf.keras.layers.LSTM(units=32)  # second lstm
    
  encoder_o = encoder_quest(encoder_embedding_out)
  encoder_output = encoder_quest_2(encoder_o)
  #encoder_states = [h, c]

  quest_output = tf.keras.layers.Dense(units=256, activation="relu")(encoder_output)

  model_question = tf.keras.Model(question_input, quest_output)

  #### Combine the two networks

  concat = tf.keras.layers.concatenate([quest_output, encoded_image], axis=1)
  #merged_model.add(tf.keras.layers.Concatenate([model_img, model_question]))
  #concat = tf.keras.layers.concatenate([decoder, encoded_image])
  combin = tf.keras.layers.Dense(units=256, activation="relu")(concat)
  classif = tf.keras.layers.Dense(units=num_classes, activation="softmax")(combin)
  #merged_model.add(Dense(1000, activation='softmax'))

  merged_model = tf.keras.Model(inputs=[image_input, question_input], outputs=classif)
  #vqa_model = Model(inputs=[image_input, question_input], outputs=output)

  return merged_model


# In[14]:


model = get_model()

model.summary()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model_name = os.path.join(ckpt_dir, "cp_10.ckpt")
model.load_weights(model_name)

now = datetime.now().strftime('%b%d_%H-%M-%S')
print(str(now))

model.fit(x=train_dataset,
          epochs=100,
          steps_per_epoch=len(train_)/bs,
          validation_data=val_dataset, 
          validation_steps=len(val_)/bs,
          callbacks=callbacks)

model_name = os.path.join(models_dir, now)

os.makedirs(model_name)

model.save(model_name)


# In[15]:


model_basename = now


# In[16]:


pred_dir = os.path.join(prediction_dir, os.path.basename(model_basename))


# In[17]:


#### Process test questions

with open(os.path.join(dataset_dir, 'test_questions.json')) as f:    #read the file
  test_questions = json.load(f)

test_dataset = []

count = 0
for key, value in test_questions.items():
  test_item = Test_Data(key, value)
  test_dataset.append(test_item)

  count += 1
  print("Created " + str(count) + " test items")


# In[18]:


results = []

for item in test_dataset:
    
    item.question = question_tokenizer.texts_to_sequences([item.question])
    item.question = pad_sequences(item.question, maxlen=max_question_length)
    
    prediction = model.predict([np.expand_dims(item.image, axis=0), item.question])
    class_pred = prediction.argmax(axis=-1)
    results.append(class_pred[0])

print(results)


# In[19]:


def create_csv(results, results_dir='./'):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')
            
file_dict = {}

for pred in range(len(results)):
    
    file_dict[test_dataset[pred].object_id] = results[pred]
    
create_csv(file_dict, prediction_dir)

