#---------------------------------Start Imports---------------------------------#
import os
import re
import random
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
#----------------------------------End Imports----------------------------------#
#----------------------------------Start Setup----------------------------------#
def log(x):
    """
    Define a numerically stable logarithm function.
    Finds the stable log of x
    param x: 

    return: numerically stable log of x
    """
    return tf.math.log(tf.maximum(x, 1e-5))
#-----------------------------------End Setup-----------------------------------#
#--------------------------Start Preprocessing Methods--------------------------#
def load_real_samples():
    """
    Method used to load in the MNIST dataset

    return: a numpy array of the MNIST dataset
    """
    # Load the MNIST Dataset
    (trainX, _), (_, _) = load_data()
    # Expand the Images to 3D
    X = np.expand_dims(trainX, axis=-1)
    # Convert from ints to floats
    X = X.astype('float32')
    # Scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


def load_anime_data(resize):
    """
    Method used to load in the anime dataset

    return: a numpy array of the anime dataset
    """
    # Find all the Image Files
    image_files = os.listdir("../../images")

    # Shuffle the Order of the Files
    random.shuffle(image_files)

    images = []
    for image_file in image_files:
        image_path = os.path.join( "../../images/", image_file)

        # Open the Image
        img = Image.open(image_path)

        # Rescale the Image to the Appropriate Size
        img = img.resize(resize)

        # Convert from ints to floats
        img = np.array(img, dtype=np.float32)

        images.append(img)

    images = np.asarray(images)

    # Convert from ints to floats
    images = images.astype('float32')

    # Scale from [0,255] to [-1,1]
    images = (images - 127.5) / 127.5

    return images

def sample_real_images(dataset, n_samples):
    """
    Randomly select some examples from the inputted dataset.
    
    param dataset: the dataset from which to take some data
    param n_samples: the number of samples to take from the dataset

    return: randomly select n_samples from the dataset and output the examples
        with the appropriate labels of 1, since the images are all real
    """
    # Generate Random Indices
    index = np.random.randint(0, dataset.shape[0], n_samples)
    # Select the Images
    x = dataset[index]
    # Generate the Labels
    y = np.ones((n_samples, 1))
    return x, y
#---------------------------End Preprocessing Methods---------------------------#
#-----------------------------Start Create Generator-----------------------------#
def create_generator(latent_dim):
    """
    Used to create a generator model
    
    param latent_dim: the size of input that the generator model will take in 

    return: the generator model
    """
	# Weight Initialization
    init = RandomNormal(stddev=0.02)

	# Define Model
    model = Sequential()
    
    # Convert point in Latent Space to a point in a Foundational 7x7x256 space
    #   that becomes the bases of our generated image.
    model.add(Dense(256*7*7, kernel_initializer=init, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((7, 7, 256)))

	# Upsample to 14x14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

	# Upsample to 28x28
    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

	# Output Image of Dimension 28x28x3
    model.add(Conv2D(3, (7,7), padding='same', kernel_initializer=init))
    model.add(Activation('tanh'))
    

    return model
#------------------------------End Create Generator------------------------------#
#---------------------------Start Create Discriminator---------------------------#
def create_discriminator(in_shape):
    """
    Used to create a discriminator model
    
    param in_shape: the size of input that the disciminator model will take in 

    return: the generator model
    """
    # Weight Initialization
    init = RandomNormal(stddev=0.02)

    # Define Model
    model = Sequential()

    # Downsample to 14x14 - Suppose Input Shape 28x28
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Downsample to 7x7 - Suppose Input Shape 28x28
    model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(1, activation='linear', kernel_initializer=init))

    # Compile Model with Mean Squared Error Loss
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return model
#----------------------------End Create Discriminator----------------------------#
#--------------------------------Start Create GAN--------------------------------#
def create_gan(generator, discriminator):
    """
    Combine a generator and disciminator model that will be used to update the
        generator.
    
    param generator: the generator model
    param discriminator: the discriminator model

    return: the GAN model
    """
    # Make the Weights in the Discriminator Not Trainable
    for layer in discriminator.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    # Connect the Generator and Discriminator
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    # Compile Model with Mean Squred Error Loss
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return model
#--------------------------------End Create GAN--------------------------------#
#--------------------------Start Image Creation Methods--------------------------#
def generate_latent_points(latent_dim, number_of_points):
    """
    Randomly generate points in the latent space as input for the generator.
    
    param latent_dim: the dimension of the latent space
    param number_of_points: the number of latent points to generate

    return: number_of_points latent_dim dimension points
    """
    # Generate random points in the latent space
    latent_points = np.random.randn(latent_dim * number_of_points)
    # Reshape into the appropriate size
    latent_points = latent_points.reshape(number_of_points, latent_dim)

    return latent_points

def generate_fake_images(generator, latent_dim, n_samples):
    """
    Use the inputted generator model to generate some fake images and labels

    param generator: the generator model
    param latent_dim: the dimension of the latent space that the generator 
        takes in
    param n_samples: the number of fake images we want to generate

    return: n_samples of fake images, with the appropriate labels of 0, since 
        the images are all fake
    """
    # Generate Points in the Latent Space
    x_input = generate_latent_points(latent_dim, n_samples)
    # Generate Fake Images given the Latent Points
    x = generator.predict(x_input)
    # Generate the Labels
    y = np.zeros((n_samples, 1))

    return x, y
#---------------------------End Image Creation Methods---------------------------#
#----------------------------Start Save Image Methods----------------------------#
def save_generator_images(epoch_num, generator, latent_dim, n_samples=100):
    """
    Generate some Fake Images using the generator and save the plot of the 
        images as well as the weights of the generator model at that time.

    param epoch_num: the epoch we are on
    param generator: the generator model
    param latent_dim: the size of the latent dimension input that the generator
        model takes in
    param n_samples: the number of fake images we want to generate
    """
    # Generate Fake Images
    x, _ = generate_fake_images(generator, latent_dim, n_samples)
    # Scale from [-1,1] to [0,1]
    x = (x + 1) / 2.0
    # Plot Images
    for i in range(10 * 10):
        # Define the size of the subplot
        plt.subplot(10, 10, 1 + i)
        # Turn off the Axis
        plt.axis('off')
        # Plot the raw pixel data
        plt.imshow(x[i, :, :, :])
    # Save the plot
    filename1 = 'generated_plot_%06d.png' % (epoch_num+1)
    plt.savefig(filename1)
    plt.close()
    # Save the weights for the generator model
    filename2 = 'model_%06d.h5' % (epoch_num+1)
    generator.save(filename2)
    print('Saved %s and %s' % (filename1, filename2))

def plot_loss_graph(discriminator_real_image_losses, discriminator_fake_image_losses, generator_losses):
    """
    Used to create and save a line graph for the loss of the GAN

    param discriminator_real_image_losses: a list of the discriminator losses on
        real images
    param discriminator_fake_image_losses: a list of the discriminator losses on
        fake images
    param generator_losses: a list of the generator losses.
    """
    plt.plot(discriminator_real_image_losses, label='dlossReal')
    plt.plot(discriminator_fake_image_losses, label='dlossFake')
    plt.plot(generator_losses, label='gloss')
    plt.legend()
    filename = 'plot_line_plot_loss.png'
    plt.savefig(filename)
    plt.close()
    print('Saved %s' % (filename))
#-----------------------------End Save Image Methods-----------------------------#
#-----------------------------Start Training Method-----------------------------#
def train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=40, n_batch=64):
    """
    Used to train a generator and discriminator, used to train a GAN.

    param generator: the generator model
    param discriminator: the discriminator model
    param gan_model: the GAN model
    param dataset: the dataset we want to train the GAN on
    param latent_dim: the dimension of the latent space that the generator takes
        as input
    param n_epochs: the number of epochs we want to train for
    param n_batch: the number of images in each batch
    """
    print("Training Started")
    # Calculate the number of batches we will have per training epoch
    number_of_batches_per_epoch = int(dataset.shape[0] / n_batch)
    # Calculate the number of training iterations needed
    number_of_iterations = number_of_batches_per_epoch * n_epochs
    # Calculate the size of half a batch of samples so we can split evenly 
    #   between half fake images and half real images
    half_batch = int(n_batch / 2)
    # Make Lists for storing loss, for plotting later
    discriminator_real_image_losses = [] 
    discriminator_fake_image_losses = [] 
    generator_losses = []

    for i in range(number_of_iterations):
        # Get Real and Fake Images
        x_real, y_real = sample_real_images(dataset, half_batch)
        x_fake, y_fake = generate_fake_images(generator, latent_dim, half_batch)

        # Train the Discriminator on the Real and Fake Images
        d_loss_real = discriminator.train_on_batch(x_real, y_real)
        d_loss_fake = discriminator.train_on_batch(x_fake, y_fake)
        
        # Train the Generator On A Batch of Fake Images Using the Discriminators Loss
        latent_points = generate_latent_points(latent_dim, n_batch)
        generator_labels = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(latent_points, generator_labels)

        # Keep Track Of Losses For Plotting Purposes
        discriminator_real_image_losses.append(d_loss_real)
        discriminator_fake_image_losses.append(d_loss_fake)
        generator_losses.append(g_loss)

        # Print out Losses Every Epoch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss_real, d_loss_fake, g_loss))

        # Generate and Save Generator Images and Weights Every Epoch
        if (i+1) % (number_of_batches_per_epoch * 1) == 0:
            save_generator_images(i, generator, latent_dim)

    # Plot the Losses on A Graph to Show Training History
    plot_loss_graph(discriminator_real_image_losses, discriminator_fake_image_losses, generator_losses)
#------------------------------End Training Method------------------------------#
#-------------------------------Start Main Method-------------------------------#
def main():
    # Killing optional CPU Driver Warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Size of the Latent Space
    latent_dim = 100
    # Create the Generator
    generator = create_generator(latent_dim)
    # Create the Discriminator
    discriminator = create_discriminator(in_shape=(28, 28, 3))
    
    # Print Model Summaries
    print(generator.summary())
    print(discriminator.summary())
    
    # Create the GAN
    gan_model = create_gan(generator, discriminator)
    
    # Load Image Data
    # dataset = load_real_samples() # Used To Load MNIST Data
    dataset = load_anime_data(resize=(28, 28)) # Used to Load Anime Data
    
    # Train the GAN
    train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=20, n_batch=64)

if __name__ == '__main__':
   main()
#--------------------------------End Main Method--------------------------------#