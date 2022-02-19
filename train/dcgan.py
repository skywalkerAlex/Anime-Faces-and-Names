import time
import os
from IPython import display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import wandb

class DCGAN(object):

    def __init__(self, batch_size, n_noise = 100, num_examples_to_generate = 9):
        super().__init__()
        self.batch_size = batch_size
        self.noise = tf.random.normal([batch_size, n_noise])
        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.seed = tf.random.normal([num_examples_to_generate, n_noise])

    def generator(self):
        print("Generating DCGAN model.")
        model = tf.keras.Sequential()
        model.add(layers.Dense(45*30*256, use_bias=False, input_shape=(100,)))
        #     model.add(layers.BatchNormalization())
        #     model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Reshape((45, 30, 256)))
        assert model.output_shape == (None, 45, 30, 256)  # Note: None is the batch size
        
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 45, 30, 128)
        
        #     model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 90, 60, 64)
        
        #     model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 180, 120, 32)
        
        #     model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 360, 240, 3)
        
        print("DCGAN model completed")
        return model

    def generator_loss(self,fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    
    def discriminator(self):
        print("DCGAN descriminator model")
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[360, 240, 3]))
        model.add(layers.LeakyReLU())
        #     model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        #     model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        #     model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        #     model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1, activation="sigmoid"))
        print(model.output_shape)
        print("DCGAN descriminator Completed!!")
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    @tf.function
    def train_step(self, images):
        print("Training step")
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            print("Generator for noise")
            generator = tf.function(self.generator())
            discriminator = tf.function(self.discriminator())
            generated_images = generator(self.noise)
            print("Discriminator for images")
            real_out = discriminator(images)
            print("Discriminator for noise")
            fake_out = discriminator(generated_images)

            gen_loss = self.generator_loss(fake_out)
            print("gen_loss :",gen_loss)
            disc_loss = self.discriminator_loss(self,real_output=real_out, fake_output=fake_out)
            print("disc_loss :",disc_loss)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    

    def train(self, dataset, epochs):
        print("Training Start...")
        for epoch in range(epochs):
            print("Epoch :", epoch)
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator(), epoch + 1, self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        with tf.compat.v1.Session() as sess:
            wandb.tensorflow.log(tf.summary.merge_all())
        
            # Generate after the final epoch
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator(), epochs, self.seed)

    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input)

        fig = plt.figure(figsize=(10, 10))
        print(predictions)
        print(predictions.shape)
        for i in range(predictions.shape[0]):
            plt.subplot(3, 3, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5 )
            plt.axis('off')

        plt.savefig('./generated_images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def optimizer_checkpoint(self):
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=self.generator(),
                                        discriminator=self.discriminator())