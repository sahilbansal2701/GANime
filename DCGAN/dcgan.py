#---------------------------------Start Imports---------------------------------#
import re
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imageio import imwrite
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
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
#------------------------------Start Preprocessing------------------------------#
def load_image_batch(dir_name, batch_size=100, shuffle_buffer_size=250000, n_threads=2, image_resize=94):
    """
    Function used to load and pre-process image files. Given a directory and a 
        batch size, returns a dataset iterator that can be queried for a 
        batch of images

    param dir_name: file path to the data set
    param batch_size: the batch size of images that will be trained on each time
    param shuffle_buffer_size: representing the number of elements from this 
        dataset from which the new dataset will sample
    param n_thread: the number of threads that will be used to fetch the data

    return: an iterator into the dataset
    """

    def load_and_process_image(file_path):
        """
        Given a file path, this function opens and decodes the image stored 
            in the file.

        param file_path: a single image
        
        return: an normalized rgb image
        """
        # Load image
        image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize the image
        image = tf.image.resize(image, [image_resize, image_resize])
        # Rescale data to range (-1, 1)
        image = (image - 0.5) * 2
        return image

    # List file names/file paths
    dir_path = dir_name + '/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path)

    # Shuffle the Dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images (in parallel)
    dataset = dataset.map(map_func=load_and_process_image, num_parallel_calls=n_threads)

    # Create batch, dropping the final one which has less than batch_size 
    #   elements and finally set to reshuffle the dataset at the end of 
    #   each iteration
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset
    return dataset
#-------------------------------End Preprocessing-------------------------------#
#-----------------------------Start Generator Model-----------------------------#
class Generator_Model(tf.keras.Model):
    def __init__(self, batch_size):
        """
        The model for the generator network is defined here. 

        param batch_size: batch_size is the size of the batch of data returned       
        """
        super(Generator_Model, self).__init__()
        self.noise_size = 100
        self.output_image_size = 64 # Actual output size is 94
        self.num_output_channels = 3
        self.learning_rate = 2e-4
        self.beta_1 = 0.5
        self.batch_size = batch_size

        self.G_optimizer =  tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

        self.architecture = [
            Conv2DTranspose(self.output_image_size*8, kernel_size=4, strides=1),
            BatchNormalization( momentum=0.1, epsilon=1e-5),
            LeakyReLU(),
            Conv2DTranspose(self.output_image_size*4, kernel_size=4, strides=2),
            BatchNormalization( momentum=0.1, epsilon=1e-5),
            LeakyReLU(),
            Conv2DTranspose(self.output_image_size*2, kernel_size=4, strides=2),
            BatchNormalization( momentum=0.1, epsilon=1e-5),
            LeakyReLU(),
            Conv2DTranspose(self.output_image_size, kernel_size=4, strides=2),
            BatchNormalization( momentum=0.1, epsilon=1e-5),
            LeakyReLU(),
            Conv2DTranspose(self.num_output_channels, kernel_size=4, strides=2, activation="tanh")
        ]
    
    def sample_noise(self, batch_size, dim):
        """
        Generate random normal noise
        
        param batch_size: integer giving the batch size of noise to generate
        param dim: integer giving the dimension of the the noise to generate

        return: Tensor containing normal noise with mean 0 and variance 1 with 
            shape [batch_size, 1, 1, dim]
        """
        return tf.random.normal(shape=[batch_size, 1, 1, dim])

    @tf.function
    def call(self):
        """
        Executes the generator model on the random noise vectors.

        return: automatically samples noise and generates
            prescaled images, output_shape=[batch_size, height, width, channel]
        """
        x = self.sample_noise(self.batch_size, self.noise_size)

        for layer in self.architecture:
            x = layer(x)

        return x

    @tf.function
    def loss_function(self, d_fake_output):
        """
        Outputs the loss given the discriminator output on the generated images.

        param d_fake_output: the discrimator output on the generated images, 
            shape=[batch_size,1]

        return: loss, the binary cross entropy loss
        """

        G_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0, reduction="auto", name="binary_crossentropy" )
        label = tf.ones(shape=(self.batch_size, 1))
        loss = G_loss(label, d_fake_output)

        return loss
#------------------------------End Generator Model------------------------------#
#---------------------------Start Discriminator Model---------------------------#
class Discriminator_Model(tf.keras.Model):
    def __init__(self, batch_size):
        super(Discriminator_Model, self).__init__()
        """
        The model for the discriminator network is defined here. 
        
        param batch_size: batch_size is the size of the batch of data passed in
        """
        self.noise_size = 100
        self.output_image_size = 64 # Actual input size is 94
        self.num_output_channels = 3
        self.learning_rate = 2e-4
        self.beta_1 = 0.5
        self.batch_size = batch_size

        self.D_optimizer =  tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

        self.architecture = [
            Conv2D(self.output_image_size, kernel_size=4, strides=2),
            LeakyReLU(),
            Conv2D(self.output_image_size*2, kernel_size=4, strides=2),
            BatchNormalization( momentum=0.1, epsilon=1e-5),
            LeakyReLU(),
            Conv2D(self.output_image_size*4, kernel_size=4, strides=2),
            BatchNormalization( momentum=0.1, epsilon=1e-5),
            LeakyReLU(),
            Conv2D(self.output_image_size*8, kernel_size=4, strides=2),
            BatchNormalization( momentum=0.1, epsilon=1e-5),
            LeakyReLU(),
            Conv2D(1, kernel_size=4, strides=1), 
            Flatten(),
            Dense(1, activation="sigmoid")
        ]

    @tf.function
    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs 
            whether it is real or fake.

        param inputs: a batch of images, 
            shape=[batch_size, height, width, channels]

        return: a batch of values indicating whether the image is real or fake, 
            shape=[batch_size, 1]
        """
        x = inputs
        for layer in self.architecture:
            x = layer(x)

        return x

    def loss_function(self, d_real_output, d_fake_output):
        """
        Outputs the discriminator loss given the discriminator model output on 
            the real and generated images.

        param d_real_output: discriminator output on the real images, 
            shape=[batch_size, 1]
        param d_fake_output: discriminator output on the generated images, 
            shape=[batch_size, 1]

        return: loss, the combined binary cross entropy loss
        """
 
        D_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0, reduction="auto", name="binary_crossentropy" )
        label_real = tf.ones(shape=(self.batch_size, 1))
        loss_real = D_loss(label_real, d_real_output)
        label_fake =  tf.zeros(shape=(self.batch_size, 1))
        loss_fake = D_loss(label_fake, d_fake_output)

        return loss_real, loss_fake 
#----------------------------End Discriminator Model----------------------------#
#--------------------------Start Plot Loss Graph Method--------------------------#
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
#--------------------------End Plot Loss Graph Method--------------------------#
#----------------------------Start Training Methods----------------------------#
def train_a_batch(x, generator_model, discriminator_model, update_discriminator):
    """
    Trains the discriminator and generator model on a single batch of images.

    param x: a batch of real preprocessed images
    param generator_model: the generator model to be trained
    param discriminator_model: the discriminator model to be trained
    params update_discriminator: boolean used to determine whether or not to 
        update the discriminator

    return: g_loss, d_loss: the generator loss and the discriminator loss on 
        the batch
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generated images
        G_sample = generator_model.call()
        
        # Find Discriminator Output on Real Images
        logits_real = discriminator_model(x)
        # Find Discriminator Output on Fake Images
        logits_fake = discriminator_model(G_sample)
        
        # Compute Losses For Both Models
        g_loss = generator_model.loss_function(logits_fake)
        d_loss_real, d_loss_fake = discriminator_model.loss_function(logits_real, logits_fake)
        combined_loss = d_loss_real + d_loss_fake

    # Call optimizer and apply gradient descent on the generator
    gradients_generator = gen_tape.gradient(g_loss, generator_model.trainable_variables)
    generator_model.G_optimizer.apply_gradients(zip(gradients_generator,generator_model.trainable_variables))
    
    # Only Gradient Descent the discriminator if the boolean update_discriminator
    #     is true.
    if(update_discriminator == True):
        gradients_discriminator = disc_tape.gradient(combined_loss, discriminator_model.trainable_variables)
        discriminator_model.D_optimizer.apply_gradients(zip(gradients_discriminator, discriminator_model.trainable_variables))
        
    return g_loss, d_loss_real, d_loss_fake

def train_gan(dataset, print_every, batch_size, num_epoch, generator_model, discriminator_model):
    """
      Train a GAN for a certain number of epochs.

      params dataset: the dataset to train the GAN on
      params print_every: prints the epoch, generator loss, and discriminator 
        loss after every print_every amount of iterations
      params batch_size: the number of images in a batch
      params num_epoch: the number of epochs to train the GAN on
      params generator_model: the generator model to be trained
      params discriminator_model: the discriminatro model to be trained
    
      returns: nothing
    """
    # Used to save the weights of the GAN
    checkpoint_dir = '../training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_model.G_optimizer,
                                 discriminator_optimizer=discriminator_model.D_optimizer,
                                 generator=generator_model,
                                 discriminator=discriminator_model)
    
    it = 0
    update_discriminator = True

    # Make Lists for storing loss, for plotting later
    discriminator_real_image_losses = [] 
    discriminator_fake_image_losses = [] 
    generator_losses = []
    
    # Trains the GAN over the amount of epochs inputted
    for e in range(num_epoch):

      # Loops through each batch of images in the dataset
      for minibatch in dataset:

        # Run a batch of data through the network
        G_loss_curr, D_loss_real, D_loss_fake  = train_a_batch(minibatch, generator_model, discriminator_model, update_discriminator)

        discriminator_real_image_losses.append(D_loss_real)
        discriminator_fake_image_losses.append(D_loss_fake)
        generator_losses.append(G_loss_curr)

        # Used to update the discriminator once every two batches
        if(update_discriminator == True):
            update_discriminator = False
        else:
            update_discriminator = True

        # Print loss every print_every epochs
        if it % print_every == 0:
            print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(e,it, D_loss_real + D_loss_fake, G_loss_curr))
        it += 1

    # Saves the weights of the model at the end of training
    checkpoint.save(file_prefix = checkpoint_prefix)

    # Plot the Losses on A Graph to Show Training History
    plot_loss_graph(discriminator_real_image_losses, discriminator_fake_image_losses, generator_losses)

    # Generates some sample images and saves 30 images
    outputs = generator_model.call()
    img = outputs[:30]
    
    # Rescale the image from (-1, 1) to (0, 255)
    img = ((img / 2) - 0.5) * 255
    # Convert to uint8
    img = img.numpy()
    img = img.astype(np.uint8)
    # Save images to disk
    for i in range(0, 30):
        img_i = img[i]
        s = '../outputimages/'+str(i)+'.png'
        imwrite(s, img_i)
#-----------------------------End Training Methods-----------------------------#
#-------------------------------Start Main Method-------------------------------#
def main():
    batch_size = 100
    num_epoch = 40
    print_every=50
    image_resize=94
    
    # Killing optional CPU Driver Warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Preprocess the Sataset
    dataset = load_image_batch("../../images", batch_size=batch_size, image_resize=image_resize)

    # Create Generator and Discriminator
    g_model = Generator_Model(batch_size=batch_size)
    d_model = Discriminator_Model(batch_size=batch_size)

    # Trains the GAN
    train_gan(dataset, print_every=print_every, batch_size=batch_size, num_epoch=num_epoch, generator_model=g_model, discriminator_model=d_model)

if __name__ == '__main__':
   main()
#--------------------------------End Main Method--------------------------------#