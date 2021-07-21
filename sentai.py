import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import os
import json
from tkinter import *
import sys
from tkinter import filedialog


#File Dialogs and Inputs
f = str(os.path.join(sys.path[0], "config.json"))
if os.path.isfile(f) and os.access(f, os.R_OK):
    print("File exists and is readable")
    with open('config.json', 'r') as b:
        config = json.load(b)
        dataPath = config['key1']
        modelPath = config['key2']
        txtcol = config['key3']
        labelcol = config['key4']
else:
    print("Either file is missing or is not readable, creating file...")
    root = Tk()
    root.withdraw()
    dataPath = filedialog.askopenfilename(initialdir="/", title="Select your CSV file",
                                             filetypes=(("CSV files", "*.csv*"), ("all files", "*.*")))
    modelPath = filedialog.askdirectory()

    txtcol = input('Text Column Name in CSV file (Case Sensitive): ')

    labelcol = input('Label Column Name in CSV file (Case Sensitive) (Only Binary is supported, labels must be 0 or 1): ')

    root.destroy()

    config = {"key1": dataPath, "key2": modelPath, "key3": txtcol, "key4": labelcol}

    with open('config.json', 'w') as t:
        json.dump(config, t)

jsonini = input("Do you want to recreate a json file? y or n: ")
if jsonini == "y":
    root = Tk()
    root.withdraw()
    dataPath = filedialog.askopenfilename(initialdir="/", title="Select your CSV file",
                                          filetypes=(("CSV files", "*.csv*"), ("all files", "*.*")))
    modelPath = filedialog.askdirectory()

    txtcol = input('Text Column Name in CSV file (Case Sensitive): ')

    labelcol = input(
        'Label Column Name in CSV file (Case Sensitive) (Only Binary is supported, labels must be 0 or 1): ')

    root.destroy()

    config = {"key1": dataPath, "key2": modelPath, "key3": txtcol, "key4": labelcol}

    with open('config.json', 'w') as t:
        json.dump(config, t)

#USE Model and proprocessing with Pandas Dataframe from Amit Chaudhary (START)

df = pd.read_csv(dataPath)

x_train, x_test, y_train, y_test = train_test_split(df[txtcol],
                                                    df[labelcol],
                                                    test_size=0.3,
                                                    stratify=df[labelcol],
                                                    random_state=42)

x = tf.keras.layers.Input(shape=[], dtype=tf.string)
y = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                   trainable=True)(x)
z = tf.keras.layers.Dense(1, activation='sigmoid')(y)
model = tf.keras.models.Model(x, z)
model.summary()

learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.001,
    end_learning_rate=0.0001,
    decay_steps=10000,
    power=0.5)
opt = keras.optimizers.Adam(learning_rate=learning_rate_fn)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          epochs=2,
          validation_data=(x_test, y_test))

#USE Model and proprocessing with Pandas Dataframe from Amit Chaudhary (END)


model.save(str(modelPath) + '/my_model.h5')
