from keras.api._v2.keras import preprocessing
import tensorflow
from tensorflow.keras.utils import img_to_array, load_img
import glob
import os
from multiprocessing import Process

datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

samples = 5


def generate(imageDirectoryName, imageLocation):
    img = load_img(imageLocation)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=imageDirectoryName,
        save_prefix="cat",
        save_format="jpg",
    ):
        i += 1
        if i > samples:
            break


def augument(person):
    personID = person.split("/")[-1]
    pictures = glob.glob(f"{person}/*")
    print(f"\nProcessing Person{personID}\n")
    for picture in pictures:
        print(f"Augumenting {picture} ....\n")
        generate(person, picture)


def processImages(directory):
    absPath = os.path.abspath(directory)
    persons = glob.glob(f"{absPath}/*")
    processCount = len(persons)

    processes = []

    for i in range(processCount):
        person = persons[i]
        k = Process(target=augument, args=(person,))
        processes.append(k)

    for process in processes:
        process.start()


processImages("Training")
