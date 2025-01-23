from PIL import Image
import os
from keras import Sequential
from keras.layers import Dense, Input
import numpy as np
import pandas as pd
import collections
from tensorflow import keras
from keras import layers
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import json

# Datensätze laden
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
num_classes = 10

print("Trainingsdaten:")
print("Anzahl der Trainingsbilder:", len(x_train))
print("Dimensionen der Trainingsbilder:", x_train.shape)
print("Anzahl der Testbilder:", len(x_test))
print("Dimensionen der Testbilder:", x_test.shape)


unique, counts = np.unique(y_train, return_counts=True)
print("Häufigkeit der Trainingsdaten:")
for label, count in zip(unique, counts):
    print(f"Label {label}: {count} Bilder")

# Beispielbild anzeigen
img_no = 10
print("Beispielbild Nummer:", img_no)
print("Label des Beispielbildes:", y_train[img_no])
plt.figure()
plt.imshow(x_train[img_no], cmap='gray')
plt.grid(False)
plt.show()

# Pixel des zehnten Bildes
print("Pixel des zehnten Bildes:")
print(x_train[10])

# Bilder speichern
for i in range(0, 100):
    im = Image.fromarray(x_train[i])
    real = y_train[i]
    im.save("tmp_Daten/%d_%d.jpeg" % (i, real))

# Ordner erstellen für jede Kategorie
categories = np.unique(y_train)
base_dir = 'categorized_images'
os.makedirs(base_dir, exist_ok=True)
for category in categories:
    os.makedirs(os.path.join(base_dir, str(category)), exist_ok=True)
# Bilder in die Ordner speichern
for i in range(len(x_train)):
    im = Image.fromarray(x_train[i])
    category = y_train[i]
    im.save(os.path.join(base_dir, str(category), f"{i}.jpeg"))

# Daten skalieren und vorbereiten
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Bilder auf (28, 28, 1) erweitern
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape, "x_train shape:")
print(x_train.shape[0], "number of train samples")
print(x_test.shape[0], "number of test samples")

nr_labels_y = collections.Counter(y_train)  # Anzahl der Labels zählen
print(nr_labels_y, "Number of labels")

# Labels in One-Hot-Encoding umwandeln
y_train = keras.utils.to_categorical(y_train, num_classes)
y_labels = y_test  # Original-Labels speichern
y_test = keras.utils.to_categorical(y_test, num_classes)

# Modell erstellen
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# Modell Trainieren
batch_size = 64
epochs = 12

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# Modell evaluieren
score = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Lernkurve zeichnen
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

# Vorhersagen auf Testdaten durchführen und vergleichen
pred = model.predict(x_test)
print(pred[1])  # Vorhersage für Bild 1
pred_1 = np.argmax(pred[1])
print(pred_1)

for i in range(0, 100):
    pred_i = np.argmax(pred[i])  # Index des höchsten Wertes ermitteln
    print(y_labels[i], pred_i)

# Modell speichern
model.save('model.h5')
model.save_weights("model.weights.h5")

weights = model.get_weights()
j = json.dumps(pd.Series(weights).to_json(orient='values'), indent=3)
print(j)

model = keras.models.load_model('model.h5')
model.load_weights("model.weights.h5")

model_json = model.to_json()
print(model_json)