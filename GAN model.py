import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random

SEED = 42
tf.random.set_seed(SEED)

class GAN:
    def __init__(self, num_features, num_historical_days, generator_input_size=200, is_train=True):
        # The generator takes in random noise and outputs data with the same structure as historical stock data
        self.generator_input_size = generator_input_size
        self.num_historical_days = num_historical_days
        self.num_features = num_features
        self.is_train = is_train

        self.build_model()

    def sample_Z(self, batch_size, n):
        return np.random.uniform(-1., 1., size=(batch_size, n))

    def build_generator(self):
        # Generator
        # Input shape: [None, generator_input_size]
        # Output shape: [None, num_historical_days * num_features]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.num_historical_days * 10, activation='sigmoid', input_shape=(self.generator_input_size,)),
            tf.keras.layers.Dense(units=self.num_historical_days * 5, activation='sigmoid'),
            tf.keras.layers.Dense(units=self.num_historical_days * self.num_features),
            tf.keras.layers.Reshape(target_shape=(self.num_historical_days, self.num_features))
        ])
        return model

    def build_discriminator(self):
        # Discriminator
        # Input shape: [None, num_historical_days, num_features]
        # Output shape: [None, 1]
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='leaky_relu', input_shape=(self.num_historical_days, self.num_features)),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='leaky_relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
        return model

    def build_model(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Placeholders for input data and random noise
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_historical_days, self.num_features], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.generator_input_size], name='Z')

        # Generator and Discriminator outputs
        self.gen_data = self.generator(self.Z)
        self.real_output = self.discriminator(self.X)
        self.fake_output = self.discriminator(self.gen_data)

        # Losses
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.ones_like(self.fake_output)))
        self.disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_output, labels=tf.ones_like(self.real_output))) + \
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.zeros_like(self.fake_output)))

        # Optimizers
        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

        # Training steps
        # Add the variables of the generator and discriminator respectively
        self.training_step_gen = self.gen_optimizer.minimize(self.gen_loss, var_list=self.generator.variables)
        self.training_step_disc = self.disc_optimizer.minimize(self.disc_loss, var_list=self.discriminator.variables)

# Training code will go here
class TrainGAN:
    def __init__(self, num_historical_days, batch_size=128):
        self.batch_size = batch_size
        self.data = []
        self.gan = GAN(num_features=5, num_historical_days=num_historical_days, generator_input_size=200)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.batch_data()

    def batch_data(self):
        files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]
        for file in files:
            df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = ((df - df.rolling(self.gan.num_historical_days).mean().shift(-self.gan.num_historical_days)) /
                  (df.rolling(self.gan.num_historical_days).max().shift(-self.gan.num_historical_days) - df.rolling(
                      self.gan.num_historical_days).min().shift(-self.gan.num_historical_days)))
            df = df.dropna()
            df = df[400:]
            for i in range(self.gan.num_historical_days, len(df), self.gan.num_historical_days):
                self.data.append(df.values[i - self.gan.num_historical_days:i])

    def train_step(self, real_data):
        noise = tf.random.normal([self.batch_size, self.gan.generator_input_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.gan.generator(noise, training=True)

            real_output = self.gan.discriminator(real_data, training=True)
            fake_output = self.gan.discriminator(generated_data, training=True)

            gen_loss = self.gan.generator_loss(fake_output)
            disc_loss = self.gan.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.gan.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.gan.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.gan.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.gan.discriminator.trainable_variables))

    def train(self, epochs, print_steps=100, save_steps=500):
        for epoch in range(epochs):
            start = 0
            for i in range(0, len(self.data), self.batch_size):
                batch_data = np.array(self.data[start:start + self.batch_size])
                start += self.batch_size
                self.train_step(batch_data)

                if i % print_steps == 0:
                    print(f'Epoch {epoch + 1}, Batch {i}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

                if (i + 1) % save_steps == 0:
                    self.gan.generator.save_weights('./gan_generator.h5')
                    self.gan.discriminator.save_weights('./gan_discriminator.h5')


if __name__ == '__main__':
    tf.keras.backend.clear_session()  # For easy reset of notebook state.
    gan_trainer = TrainGAN(num_historical_days=30, batch_size=128)
    gan_trainer.train(epochs=5000)



# Instantiate the GAN model
num_features = 5
num_historical_days = 30 # This should match your dataset's historical days
gan = GAN(num_features=num_features, num_historical_days=num_historical_days)

