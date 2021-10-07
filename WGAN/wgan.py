#---------------------------------Start Imports---------------------------------#
import os, sys
import numpy as np
from PIL import Image
from matplotlib import pyplot
from numpy import mean, random
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
#----------------------------------End Imports----------------------------------#
#--------------------------Start Preprocessing Methods--------------------------#
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
      image_path = os.path.join("../../images/", image_file)

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

   # select real samples
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
   # Generate the Labels, -1 for Real Images
   y = -np.ones((n_samples, 1))
   return x, y
#---------------------------End Preprocessing Methods---------------------------#
#---------------------------Start Wasserstein Methods---------------------------#
def wasserstein_loss(y_true, y_pred):
   """
   Method used to calculate the Wasserstein loss

   param y_true: the true labels
   param y_pred: the predictions

   return: the wasserstein loss
   """
   return backend.mean(y_true * y_pred)

class ClipConstraint(Constraint):
   """
   Class used to add a constraint on the model so that we can clip the model's
   weights so that we satisfy the constraints needed to use the Wasserstein
   distance formula
   """

   def __init__(self, clip_value):
      self.clip_value = clip_value
 
   def __call__(self, weights):
      """
      Clip the weights

      param weights: the weights to clip

      return: the weights after being clipped
      """
      return backend.clip(weights, -self.clip_value, self.clip_value)
 
   def get_config(self):
      """
      The configuration of the constraint

      return: the clip value
      """
      return {'clip_value': self.clip_value}
#----------------------------End Wasserstein Methods----------------------------#
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

   # Convert point in Latent Space to a point in a Foundational 7x7x128 space
   #   that becomes the bases of our generated image.
   model.add(Dense(128 * 7 * 7, kernel_initializer=init, input_dim=latent_dim))
   model.add(LeakyReLU(alpha=0.2))
   model.add(Reshape((7, 7, 128)))

   # Upsample to 14x14
   model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(BatchNormalization())
   model.add(LeakyReLU(alpha=0.2))

   # Upsample to 28x28
   model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(BatchNormalization())
   model.add(LeakyReLU(alpha=0.2))

   # Output Image of Dimension 28x28x3
   model.add(Conv2D(3, (7,7), activation='tanh', padding='same', kernel_initializer=init))

   return model
#------------------------------End Create Generator------------------------------#
#------------------------------Start Create Critic------------------------------#
def create_critic(in_shape):
   """
   Used to create a critic model

   param in_shape: the size of input that the critic model will take in 

   return: the critic model
   """
   # Weight Initialization
   init = RandomNormal(stddev=0.02)

   # Weight Constraint
   const = ClipConstraint(0.01)
   # Define Model
   model = Sequential()

   # Downsample to 14x14 - Suppose Input Shape 28x28
   model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
   model.add(BatchNormalization())
   model.add(LeakyReLU(alpha=0.2))

   # Downsample to 7x7 - Suppose Input Shape 28x28
   model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
   model.add(BatchNormalization())
   model.add(LeakyReLU(alpha=0.2))

   # Scoring
   model.add(Flatten())
   model.add(Dense(1, activation='linear'))

   # Compile Model with Wasserstein Loss and RMSprop
   opt = RMSprop(lr=0.00005)
   model.compile(loss=wasserstein_loss, optimizer=opt)

   return model
#-------------------------------End Create Critic-------------------------------#
#--------------------------------Start Create GAN--------------------------------#
# define the combined generator and critic model, for updating the generator
def create_gan(generator, critic):
   """
   Combine a generator and critic model that will be used to update the
   generator.
      
   param generator: the generator model
   param critc: the critic model

   return: the GAN model
   """
   # Make the Weights in the Critic Not Trainable
   for layer in critic.layers:
      if not isinstance(layer, BatchNormalization):
         layer.trainable = False

   # Connect the Generator and Critic
   model = Sequential()
   model.add(generator)
   model.add(critic)

   # Compile Model
   opt = RMSprop(lr=0.00005)
   model.compile(loss=wasserstein_loss, optimizer=opt)

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
 
# use the generator to generate n fake examples, with class labels
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
   y = np.ones((n_samples, 1))

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
      pyplot.subplot(10, 10, 1 + i)
      # Turn off the Axis
      pyplot.axis('off')
      # Plot the raw pixel data
      pyplot.imshow(x[i, :, :, :])
   # Save the plot
   filename1 = 'generated_plot_%04d.png' % (epoch_num+1)
   pyplot.savefig(filename1)
   pyplot.close()
   # Save the weights for the generator model
   filename2 = 'model_%04d.h5' % (epoch_num+1)
   generator.save(filename2)
   print('>Saved: %s and %s' % (filename1, filename2))
 
def plot_loss_graph(d1_hist, d2_hist, g_hist):
   """
   Used to create and save a line graph for the loss of the GAN

   param critic_real_image_losses: a list of the critic losses on
      real images
   param critic_fake_image_losses: a list of the critic losses on
      fake images
   param generator_losses: a list of the generator losses.
   """
   pyplot.plot(d1_hist, label='crit_real')
   pyplot.plot(d2_hist, label='crit_fake')
   pyplot.plot(g_hist, label='gen')
   pyplot.legend()
   filename = 'plot_line_plot_loss.png'
   pyplot.savefig(filename)
   pyplot.close()
   print('Saved %s' % (filename))
#-----------------------------End Save Image Methods-----------------------------#
#-----------------------------Start Training Method-----------------------------#
def train(generator, critic, gan_model, dataset, latent_dim, n_epochs=30, n_batch=64, n_critic=5):
   """
   Used to train a generator and critic, used to train a GAN.

   param generator: the generator model
   param critic: the critic model
   param gan_model: the GAN model
   param dataset: the dataset we want to train the GAN on
   param latent_dim: the dimension of the latent space that the generator takes
      as input
   param n_epochs: the number of epochs we want to train for
   param n_batch: the number of images in each batch
   param n_critic: the number of times to update the critic for each update to
      the generator
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
   critic_loss_real = [] 
   critic_loss_fake = []
   g_hist = []

   for i in range(number_of_iterations):
      # Update the Critic more than the Generator
      c_loss_r = []
      c_loss_f = []
      for _ in range(n_critic):
         # Get Real and Fake Images
         X_real, y_real = sample_real_images(dataset, half_batch)
         X_fake, y_fake = generate_fake_images(generator, latent_dim, half_batch)

         # Train the Critic on the Real and Fake Images
         c_lossr = critic.train_on_batch(X_real, y_real)
         c_lossf = critic.train_on_batch(X_fake, y_fake)

         # Keep Track Of Losses For Plotting Purposes
         c_loss_r.append(c_lossr)
         c_loss_f.append(c_lossf)

      # Train the Generator On A Batch of Fake Images Using the Critics Loss
      x_gan = generate_latent_points(latent_dim, n_batch)
      y_gan = -np.ones((n_batch, 1))
      g_loss = gan_model.train_on_batch(x_gan, y_gan)

      # Keep Track Of Losses For Plotting Purposes
      critic_loss_real.append(mean(c_loss_r))
      critic_loss_fake.append(mean(c_loss_f))
      g_hist.append(g_loss)

      # Print out Losses Every Epoch
      print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, critic_loss_real[-1], critic_loss_fake[-1], g_loss))

      # Generate and Save Generator Images and Weights Every Epoch
      if (i+1) % number_of_batches_per_epoch == 0:
         save_generator_images(i, generator, latent_dim)

   # Plot the Losses on A Graph to Show Training History
   plot_loss_graph(critic_loss_real, critic_loss_fake, g_hist)
#------------------------------End Training Method------------------------------#
#-------------------------------Start Main Method-------------------------------#
def main():
   # Killing optional CPU Driver Warnings
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

   # Size of the Latent Space
   latent_dim = 50
   # Create the Generator
   generator = create_generator(latent_dim)
   # Create the Critic
   critic = create_critic(in_shape=(28,28,3))

   # Print Model Summaries
   print(generator.summary())
   print(critic.summary())

   # Create the GAN
   gan_model = create_gan(generator, critic)

   # Load Image Data
   dataset = load_anime_data(resize=(28, 28))

   # Train the GAN
   train(generator, critic, gan_model, dataset, latent_dim, n_epochs=30, n_batch=64, n_critic=5)


if __name__ == '__main__':
   main()
#--------------------------------End Main Method--------------------------------#