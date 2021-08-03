import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Reshape, LeakyReLU, Convolution2D, UpSampling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
import random
import numpy as np
import pickle

# NN
class NeuralNetwork:
    """This outlines a very rudimentary neural network"""

    def __init__(self):
        """Initializes some variables"""
        self.model = Sequential()
        self.any_hidden_layers = False

        self.input_shape = 'n/a'

        self.x_train = 'n/a'
        self.y_train = 'n/a'
        self.x_test = 'n/a'
        self.y_test = 'n/a'

    def input_data(self, X='n/a', y='n/a'):
        """
        Input your data here.



        Data is for training purposes only
          Evaluating testing data must be done through the 'evaluate_accuracy' function
        """
        if (X == 'n/a' or y == 'n/a'):
            sys.exit('no data was given')
        else:
            self.x_train = X
            self.y_train = y
            self.input_shape = X[0].shape

    def add_input_layer(self, input_shape='n/a'):
        """
        Adds an input layer to your model.



        If you have already executed the 'input_data' command, the input_shape will be automatically calculated.
          You can override this by manually entering a value for 'input_shape'.
        """
        if (self.input_shape == 'n/a'):
            if (input_shape == 'n/a'):
                sys.exit("define the input shape of your data or use the 'input_data' function")
            else:
                self.input_shape = input_shape
        else:
            if (input_shape == 'n/a'):
                pass
            else:
                self.input_shape = input_shape

    def add_hidden_layer(self, neurons=100, activation_func='relu'):
        """
        Adds a hidden layer to your model



        The number of neurons for each layer defaults to 100
          Override this by passing (neurons = x) where x is the number of neurons you want.

        The activation function for each layer defaults to 'relu'
          You will have access to all other activation functions present in the 'keras' library.
          For example, passing the parameter (activation_func = 'sigmoid') will add a layer
          that uses the 'sigmoid' activation function.

        """
        if (self.any_hidden_layers == False):
            self.model.add(Dense(neurons, activation=activation_func, input_shape=self.input_shape))
            self.any_hidden_layers = True
        else:
            self.model.add(Dense(neurons, activation=activation_func))

    def add_output_layer(self, task='classify', neurons=0, activation_func='n/a'):
        """
        Adds the output layer for a neural network



        You will need to input the number of ouput-neurons needed for your model by
        passing (neurons = x) where x is the number of neurons you need.

        The task for the neural network defaults to 'classify'.
          For regression problems, pass (task = 'regressor') when calling the function

        Activation function for 'classify' is defaulted to 'softmax'. Activation function
        for 'regressor' is defaults to 'relu'.
          These can be changed by manually setting (activation_func = 'relu') when
          calling your function.
        """

        task = task.lower()
        activation_func = activation_func.lower()

        if (task == 'regressor' and activation_func == 'n/a'):
            activation_func = 'relu'
        if (task == 'classify' and activation_func == 'n/a'):
            activation_func = 'softmax'

        if (self.any_hidden_layers == False):
            self.model.add(Dense(neurons, activation=activation_func, input_shape=self.input_shape))
        else:
            self.model.add(Dense(neurons, activation=activation_func))

    def run_model(self, X='n/a', y='n/a', epochs=10, batch_size=64, shuffle=True, optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy']):
        """
        Compiles and trains the neural network



        The training data can either be passed in through the 'input_data' function or
        directly through this 'runmodel' function.
          There is no need to pass in your data twice.

          If you pass in data through this function, it is automatically assumed to be
          training data. To split the data, see the description for 'input_data'

        The number of training iterations (epochs) is defaulted to 10.
          Override this by entering (epochs = x) where x is the number of epochs you need

        'optimizer', 'loss', and 'metrics' are defaulted to 'Adam', 'categorical_crossentropy',
        and ['accuracy'] respectively.
          These can be altered to other values defined in the 'keras' library
        """

        if (X == 'n/a' or y == 'n/a'):
            if (self.x_train == 'n/a' or self.y_train == 'n/a'):
                sys.exit('missing training data')
        else:
            self.x_train = X
            self.y_train = y
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        self.model.fit(
            x=self.x_train,
            y=to_categorical(self.y_train),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def create(self, task='classify', output_neurons=0, epochs=10, X='n/a', y='n/a'):
        """
        Creates a rudimentary neural network that will work for most simple tasks
          There is 1 input layer, 2 hidden layers, and 1 output layer



        The number of neurons in the ouput layer must be entered.
          Exception: number of output_neurons defaults to 1 if you specify (task = 'regressor')
          instead of 'classify'

        You can also input your data through this function so that you can avoid
        using the 'input_data' function
          Data only needs to be entered once

        Number of epochs is defaulted to 10
        """
        self.input_data(X=X, y=y)
        self.add_input_layer(self.input_shape)
        self.add_hidden_layer()
        self.add_hidden_layer()
        task = task.lower()
        if (task == 'regressor' and output_neurons == 0):
            output_neurons = 1
            self.add_output_layer(task='regressor', neurons=output_neurons)
        elif (task == 'classify' and output_neurons == 0):
            sys.exit('specify the number of output neurons for your data')
        self.add_output_layer(task=task, neurons=output_neurons)

    def create_and_run(self, task='classify', output_neurons=0, epochs=10, X='n/a', y='n/a', batch_size=64):
        """
        This function creates and runs a simple neural network



        To use this function, you must:
          1. Pass in your data unless you have already run the 'input_data' function

          2. Define the number of output neurons in the network
              You can skip this step if you set (task = 'regressor')

        """
        self.create(task=task, output_neurons=output_neurons, epochs=epochs, X=X, y=y)
        self.run_model(epochs=epochs, batch_size=batch_size)

    def evaluate_accuracy(self, X='n/a', y='n/a', display_accuracy=True):
        """
        Evaluates the performance of your model using testing data based on accuracy



        Testing accuracy is displayed by default
          You can turn this off by setting (display_accuracy = False)
        """
        if (X == 'n/a' or y == 'n/a'):
            if (self.x_test == 'n/a' or self.y_test == 'n/a'):
                print("Missing testing data")
        else:
            self.x_test = X
            self.y_test = y

        self.test_loss, self.test_accuracy = self.model.evaluate(self.x_test, to_categorical(self.y_test))

        if (display_accuracy == True):
            print("Testing accuracy: ", self.test_accuracy)

        return self.test_accuracy

    def predict(self, X='n/a'):
        """
        Makes a prediction for some data



        The network must have been trained by now
        """
        if (X == 'n/a'):
            sys.exit("no data was given")

        return self.model.predict(X)

    def save_model(self, path='n/a'):
        """
        Saves to the neural network to 'path'
          Can be loaded into a new model through the 'load_model' function
        """
        if (path == 'n/a'):
            sys.exit('no path was given')
        self.model.save(path)
        print("model was saved to: ", path)

    def load_model(path):
        """
        Loads a model from a given path

        The following code shows how to load a model into
        the variable 'loaded_model':
          loaded_model = load_model('path')

        """
        return (tensorflow.keras.models.load_model(path))


# CNN
class CNN:
    def __init__(self, num_class, train_data=None, train_labels=None, test_data=None, test_labels=None, type="custom",
                 model=None, from_generator=False):

        if from_generator:
            self.train_data = train_data
            self.test_data = test_data
            self.num_class = num_class
            self.STEPS_PER_EPOCH = train_data.n // train_data.batch_size
            self.VALIDATION_STEPS = test_data.n // test_data.batch_size
            self.from_generator = True
            self.input_shape = train_data.image_shape

        else:
            self.train_data = train_data
            self.train_labels = train_labels
            self.test_data = test_data
            self.test_labels = test_labels
            self.num_class = num_class
            self.from_generator = False
            self.input_shape = train_data.shape[1:]

        self.type = type

        if type == "resnet50":
            base_model = tensorflow.keras.applications.resnet50.ResNet50(input_shape=self.input_shape, weights="imagenet",
                                                              include_top=False)
            x = tensorflow.keras.layers.GlobalAveragePooling2D()(base_model.output)
            output = None
            if num_class > 2:
                output = Dense(num_class, activation='softmax')(x)
            else:
                output = Dense(1, activation='sigmoid')(x)
            self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=output)
        
        elif type == "resnet101":
            base_model = tensorflow.keras.applications.ResNet101(input_shape=self.input_shape, weights="imagenet",
                                                              include_top=False)
            x = tensorflow.keras.layers.Flatten()(base_model.output)
            output = None
            if num_class > 2:
                output = Dense(num_class, activation='softmax')(x)
            else:
                output = Dense(1, activation='sigmoid')(x)
            self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=output)
        

        elif type == "inception_resnet":
            base_model = tensorflow.keras.applications.InceptionResNetV2(input_shape=self.input_shape, weights="imagenet",
                                                              include_top=False)
            x = tensorflow.keras.layers.Flatten()(base_model.output)
            output = None
            if num_class > 2:
                output = Dense(num_class, activation='softmax')(x)
            else:
                output = Dense(1, activation='sigmoid')(x)
            self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=output)
        
        elif type == "inception":
            base_model = tensorflow.keras.applications.InceptionV3(input_shape=self.input_shape, weights="imagenet",
                                                              include_top=False)
            x = tensorflow.keras.layers.Flatten()(base_model.output)
            output = None
            if num_class > 2:
                output = Dense(num_class, activation='softmax')(x)
            else:
                output = Dense(1, activation='sigmoid')(x)
            self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=output)

        elif type == "densenet121":
            base_model = tensorflow.keras.applications.DenseNet121(input_shape=self.input_shape, weights="imagenet",
                                                              include_top=False)
            x = tensorflow.keras.layers.Flatten()(base_model.output)
            output = None
            if num_class > 2:
                output = Dense(num_class, activation='softmax')(x)
            else:
                output = Dense(1, activation='sigmoid')(x)
            self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=output)
        elif type == "vgg":

            base_model = tensorflow.keras.applications.VGG19(input_shape=self.input_shape, weights="imagenet",
                                                  include_top=False)

            for layer in base_model.layers:
                layer.trainable = False

            x = Flatten()(base_model.output)
            output = None

            if num_class > 2:
                output = Dense(num_class, activation='softmax')(x)
            else:
                output = Dense(1, activation='sigmoid')(x)

            self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=output)
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (2, 2), padding='same', input_shape=train_data.shape[1:]))
            self.model.add(Activation('relu'))

    def addLayer(self, type, out_activation='softmax', conv_filter=64):

        if self.type != 'custom':
            return

        if type == 'conv' or type == 'convolution':

            self.model.add(Conv2D(conv_filter, (5, 5)))
            self.model.add(Activation('relu'))

        elif type == 'pool' or type == 'pooling':

            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        elif type == 'fc' or type == 'fully connected':

            self.model.add(Flatten())
            self.model.add(Dense(512))
            self.model.add(Activation('relu'))

        elif type == 'out' or type == 'output':

            self.model.add(Dense(self.num_class))
            self.model.add(Activation(out_activation))

    def compute(self, loss, train=True, optimizers="adam", lr=0.001, batch=32, num_epochs=10):
        if train:
            if optimizers == "adam":
                self.model.compile(loss=loss, optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

            elif optimizers == "sgd":
                self.model.compile(loss=loss, optimizer=tensorflow.keras.optimizers.SGD(learning_rate=lr), metrics=['accuracy'])

            if self.from_generator:
                self.model.fit(self.train_data, validation_data=self.test_data,
                               steps_per_epoch=self.STEPS_PER_EPOCH, validation_steps=self.VALIDATION_STEPS,
                               epochs=num_epochs)
            else:
                self.model.fit(self.train_data, self.train_labels, batch_size=batch, epochs=num_epochs)
        if self.from_generator:
            return self.model.evaluate_generator(self.test_data, verbose=0)
        else:
            return self.model.evaluate(self.test_data, self.test_labels, verbose=0)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def save(self, name="model"):
        self.model.save(f"{name}.h5")

    def load(self, path):
        model = tensorflow.keras.models.load_model(path)


# GAN
class GAN:
    def __init__(self, train_data, inDim):

        if len(train_data.shape) == 3:
            train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))

        # input dimension for generator
        self.inDim = inDim
        self.train_data = train_data

        # number of pixels for completley flat image
        total = 1

        for i in range(1, len(train_data.shape)):
            total *= train_data.shape[i]

        # first layer for generator
        self.gen = Sequential()
        self.gen.add(Dense(256, input_dim=self.inDim, kernel_initializer=tensorflow.keras.initializers.RandomNormal(stddev=0.02)))

        # first layer for disc
        self.disc = Sequential()
        self.disc.add(Dense(1024, input_dim=total, kernel_initializer=tensorflow.keras.initializers.RandomNormal(stddev=0.02)))

    def adddenselayer(self, network, num_neurons=512, activation='leakyrelu'):

        if activation == 'leakyrelu':
            activation = LeakyReLU(0.2)
        else:
            activation = Activation(activation)

        if network == 'g' or network == 'gen' or network == 'generator':
            self.gen.add(Dense(num_neurons, activation=activation))
        elif network == 'd' or network == 'disc' or network == 'discriminator':
            self.disc.add(Dense(num_neurons, activation=activation))

    def adddropout(self, network, rate=0.7):
        if network == 'g' or network == 'gen' or network == 'generator':
            self.gen.add(Dropout(rate))
        elif network == 'd' or network == 'disc' or network == 'discriminator':
            self.disc.add(Dropout(rate))

    def train(self, optimizer='adam', epochs=10, batch_size=128, lr=0.001):
        if optimizer == 'adam':
            optimizer = tensorflow.keras.optimizers.Adam(lr=lr, beta_1=0.5)
        elif optimizer == 'sgd':
            optimizer = tensorflow.keras.optimizers.SGD(learning_rate=lr)

        self.gen.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer)
        print(self.gen.summary())
        self.disc.trainable = False

        gan = Sequential()
        gan.add(self.gen)
        gan.add(self.disc)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        print(gan.summary())
        for i in range(epochs):
            # input for gen
            noise = np.random.randn(batch_size, self.inDim)
            # random images from gen
            genImg = self.gen.predict(noise)

            # taking a random batch of images from training data and storing in rbatch
            s1 = self.train_data.shape[0]
            idx = np.arange(s1)
            np.random.shuffle(idx)
            rBatch = self.train_data[idx[:batch_size], :]

            rBatch = np.reshape(rBatch, (
            batch_size, self.train_data.shape[1] * self.train_data.shape[2] * self.train_data.shape[3]))

            # input is a combination of generated images and random sample
            input = np.concatenate([rBatch, genImg])
            # labels are consists of 1 for random sample and 0 for generated images
            labels = np.zeros(batch_size * 2)
            labels[:batch_size] = 1

            self.disc.trainable = True
            self.disc.train_on_batch(input, labels)

            noise = np.random.randn(batch_size, self.inDim)
            labels = np.ones(batch_size)
            self.disc.trainable = False
            gan.train_on_batch(noise, labels)

    def addconvlayer(self, network, conv_filter=64, filter_size=(3, 3), stride=(1, 1)):
        if network == 'g' or network == 'gen' or network == 'generator':
            self.gen.add(Convolution2D(conv_filter, filter_size=filter_size, strides=stride))
        elif network == 'd' or network == 'disc' or network == 'discriminator':
            self.disc.add(Convolution2D(conv_filter, filter_size=filter_size, strides=stride))

    def addupsampling(self, size=(2, 2)):
        self.gen.add(tensorflow.keras.layers.UpSampling2D(size=size))

    def generate(self, num_img=1):
        noise = np.random.randn(num_img, self.inDim)
        genImg = self.gen.predict(noise)
        return genImg