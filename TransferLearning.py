import datetime
import multiprocessing
from os.path import join
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16,ResNet50V2
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.models import Model, load_model
import wx

from Config import batch_size, epochs, sessions, fixed_size, train_labels, train_path, test_path,\
    valid_path, model_name,ResNet_path, VGG_path

# ---------------------------------------------------------------
save_path = 'saved_files'
home = Path.cwd()
real_pic='apfel.jpg'

"""
This bit here is used for training on a PC with strong gpu.
Using GPU makes trainin *much* more faster
Unfortunatley there is no such luxury on my laptop
(at least it's not enough to make it work)
I might give this code to someone with really strong PC to train one of my models later
"""
#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
#config.gpu_options.allow_growth = True #cpu
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)

def import_data():
    """
    In this module I use a technique of Image Augmentation called Image Data Generators,
    this function configures them
    """
    # this is the augmentation configuration i used for training

    print("Importing data")
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # rescale pixel values from 0-255 to 0-1 so the data would be normalized
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # sub-folders and indefinitely generate batches of augmented image data
    train_generator = datagen.flow_from_directory(
        join(home, train_path),  # this is the target directory
        target_size=fixed_size,  # all Images will be resized to fixed_size
        batch_size=batch_size,
        class_mode='sparse',
    )
    # since i use sparse_categorical_crossentropy loss, i need sparse labels

    # this is a similar generator, for validation data
    validation_generator = val_datagen.flow_from_directory(
        join(home, valid_path),
        target_size=fixed_size,
        batch_size=batch_size,
        class_mode='sparse',
    )
    return train_generator, validation_generator

def build_model_ResNet():
    """
    This module uses the notion of Transfer-Learning
    I found VGG16 and ResNet models to perform best
    This function configures our model, freezes the model and adds a small module on top of it
    """
    pretrained_model = ResNet50V2(input_shape=(fixed_size[0], fixed_size[1], 3), weights='imagenet', include_top=False)
    # No need to train the imported layers.
    for layer in pretrained_model.layers:
        layer.trainable = False
    x = pretrained_model.output
    x = Flatten()(x)

    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(len(train_labels), activation='softmax')(x)
    transfer_learning_model = Model(inputs=pretrained_model.input, outputs=x)
    # just for summary in console, providing detailed information about model structure
    transfer_learning_model.summary()
    opt = Adam(learning_rate=.0003)
    transfer_learning_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return transfer_learning_model

def build_model_VGG():
    """
    This module uses the notion of Transfer-Learning
    I found VGG16 and ResNet models to perform best
    This function configures our model, freezes the model and adds a small module on top of it
    """
    pretrained_model = VGG16(input_shape=(fixed_size[0], fixed_size[1], 3), weights='imagenet', include_top=False)
    # No need to train the imported layers.
    for layer in pretrained_model.layers:
        layer.trainable = False
    x = pretrained_model.output
    x = Flatten()(x)

    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(len(train_labels), activation='softmax')(x)
    transfer_learning_model = Model(inputs=pretrained_model.input, outputs=x)
    # just for summary in console, providing detailed information about model structure
    transfer_learning_model.summary()
    opt = Adam(learning_rate=.0003)
    transfer_learning_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return transfer_learning_model


def train_model(train_generator, validation_generator):
    """
    Trains the model, requires train/val generators.
    A model with best accuracy will be stored as a file separately in the saved_files folder
    """
    # building a test generator to benchmark the model on unseen data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # how to process each class separately?
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=batch_size)
    model = build_model_VGG()

    #assembling filepath for saving checkpoints
    #is needed for tensorboard
    filepath = join(save_path, VGG_path)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',save_weights_only=True, save_best_only=True,
                                 mode='max',verbose=1)

    #setting of early stopping conditions
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=epochs // 5, verbose=1, restore_best_weights=True)

    #assembling path for logs needed for tensorboard display
    log_dir = join(home, save_path, 'logs', 'fit_smart', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [early_stopping, checkpoint, tensorboard_callback]
    # origin [sessions] models each [epochs] times
    max_acc = 0.0
    for i in range(sessions):
        # model training and evaluation
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size)
        model.load_weights(join(save_path, VGG_path))
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        # save model if it performed better
        if test_acc > max_acc:
            max_acc = test_acc
            model.save(join(home, save_path, model_name))
        print("accuracy: ", test_acc, "\n Loss:", test_loss)

def score(filepath, filename, model):
    """
    Imports a pre-trained model, feeds (filepath/filename) to the neural network
     and predicts class with confidence
    """
    print("Entering Score")
    # Pillow library is used since I open a new file that wasn't in the test folder
    img = Image.open(join(filepath, filename))
    img = img.resize(fixed_size)
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, fixed_size[0], fixed_size[1], 3)
    p = model.predict(img).tolist()[0]
    print(p)
    result = {'label': train_labels[p.index(max(p))], 'confidence': max(p)}
    return result

def display_activation(model, activations, name, col_size, row_size, act_index):
    """
    Plots activations of an image fed to the model
    Useful for visualizing features the model is picking
    Used for visualize function
    """
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='autumn')
            activation_index += 1
    fig.tight_layout(pad=1.6)
    fig.suptitle(name + ", Layer " + str(model.layers[act_index].name))
    plt.show()


def test_log(model):
    """
    A verbose log of test evaluation of the model
    """
    # Import the test data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=1)
    # Get the simple test
    print(model.evaluate(test_generator, steps=len(test_generator)))
    # Detailed test
    PR = ([], [], [])
    for i in range(test_generator.samples):
        x, y = test_generator._get_batches_of_transformed_samples([i])
        filepath = test_generator.filepaths[i]
        p = model.predict(x, ).tolist()[0]
        PR[int(y[0])].append(int(y[0]) == p.index(max(p)))
        print("prediction - ", train_labels[p.index(max(p))], " | real - ", train_labels[int(y[0])], "| confidence - ", max(p),
              "| f:", filepath)
    for i in range(3):
        print(train_labels[i], ": ", PR[i].count(True), "/", len(PR[i]), "correct - ",
              (PR[i].count(True) / len(PR[i]) * 100),
              "accuracy")

def calc_activations(model, index):
    """
    Calculates activations for a single image and outputs the calculations and the filename of the image
    Used for display_activation
    """
    # Going through each layer
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
    # Import a single image
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=fixed_size,
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=1)

    # get image from generator using index -  index used to retrieve the filename
    # the "_" used here to ignore the returning index, which I already have
    x, _ = test_generator._get_batches_of_transformed_samples([index])
    filename = test_generator.filenames[index]
    activations = activation_model.predict(x)

    return activations, filename

def visualize(model):
    """
    An interactive function which plots the features that layers of the model picked up
    To use, simply press enter to get a new image; enter the number of convolutional layer to visualize it
    enter -1 to advance to another image, finally press q after -1 to exit. enter -2 to see another class
    """
    index = 0
    current_label = train_labels[0]
    print('start')
    while str(input()) != 'q':
        print('enter')
        activations, name = calc_activations(model, index)
        layer_num = int(input())
        while layer_num != -1:
            if layer_num == -2:
                tmp = current_label
                while current_label != train_labels[train_labels.index(tmp)+1]:
                    index += 0
                    _, name = calc_activations(model, index)
                    if name.split('\\')[0] == train_labels[train_labels.index(current_label)+1]:
                        current_label = train_labels[train_labels.index(current_label) + 1]
                break
            try:
                display_activation(model, activations, name, 8, 4, layer_num)
            except Exception as e:
                print("failed - " + str(e))
            layer_num = int(input())
        index += 1


def activateResNet():

    model = build_model_ResNet()
    model.load_weights(join(save_path, ResNet_path))


    print(score(join(sys.path[0], 'real_cases'), real_pic, model))

    visualize(model)

def activateVGG():

    model = build_model_VGG()
    model.load_weights(join(save_path, VGG_path))


    print(score(join(sys.path[0], 'real_cases'), real_pic, model))
    visualize(model)

#def mainFunc():
    #model = build_model()
    #model = load_model('saved_files/vgg16.h5') #with local version

    # train, val = import_data()
    # train_model(train, val)

    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    #
    # test_generator = test_datagen.flow_from_directory( #how to process each class separately?
    #     test_path,
    #     target_size=(200, 200),
    #     color_mode="rgb",
    #     shuffle=True,
    #     class_mode='sparse',
    #     batch_size=batch_size)
    #
    #model.load_weights(join(save_path, weights_path))
    #model.save('saved_files/vgg16.h5',save_format='h5')
    #model.save('saved_files/resnet50V2.h5',save_format='h5')
    # test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

    #print(score(join(sys.path[0], 'real_cases'), 'apfel.jpg', model))
    #visualize(model)

