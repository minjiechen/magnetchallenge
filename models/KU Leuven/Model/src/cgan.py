# cGAN.py
import time
import tensorflow as tf
import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random

# dataset
import os
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import glob
import scipy.io
from tqdm import tqdm 
# Define a simple learning rate schedule
def lr_schedule(epoch):
    # lr scheduler: 0.02 * 0.95^(3000/30) = 0.000118410584
    initial_lr = 0.04
    decay_factor = 0.95
    return initial_lr * decay_factor ** (epoch/30)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


class cGAN:
    def __init__(self, material_type="A",
                 LOAD_WEIGHT_FROM_H5=True,
                 USE_HUBER_LOSS_GENERATOR=True,
                 interim_path=".\\..\\data\\interim",
                 checkpoint_path=".\\..\\data\\cGANcheckpoint",
                 train_mode = True):
        # Set the random seed for reproducibility
        seed = 42

        # Set random seed for Python's random module
        random.seed(seed)

        # Set random seed for NumPy
        np.random.seed(seed)

        # Set random seed for TensorFlow
        tf.random.set_seed(seed)

        # Disable GPU parallelism for deterministic results
        tf.config.experimental.set_visible_devices([], 'GPU')  

        # Params Discriminator
        self.lx_D=3
        self.ly_D=3
        self.ld_D=5
        self.ux_D=256
        self.uy_D=256
        self.ud_D=256

        # Material Type Parameter
        self.material = material_type

        # Training hyperparameters
        self.epochs = 3000
        self.batch_size = 32
        self.learning_rate_discriminator = 0.0001
        self.learning_rate_generator = 0.0001
        self.visualization_interval = 100
        self.LOAD_WEIGHT_FROM_H5 = LOAD_WEIGHT_FROM_H5
        self.USE_HUBER_LOSS_GENERATOR = USE_HUBER_LOSS_GENERATOR

        self.interim_path = interim_path
        self.checkpoint_path = checkpoint_path
        self.generator_h5_path = os.path.join(self.checkpoint_path,f"generator_model_material_{self.material}.h5")
        self.discriminator_h5_path = os.path.join(self.checkpoint_path,f"discriminator_model_material_{self.material}.h5")
        self.training_data_mat_path = os.path.join(self.interim_path,f'{self.material}_training.mat')
        self.target_data_mat_path = os.path.join(self.interim_path,f'{self.material}_measurements.mat')
        self.testing_data_mat_path = os.path.join(self.interim_path,f'{self.material}_testing.mat')
        
        self.train_mode = train_mode

        # Build models
        # Initialize models
        if not self.LOAD_WEIGHT_FROM_H5:
            # Initialize discriminator with specific layer structure
            print("initialization of model by random weights")
            self.lx_D = 3  # Number of layers for x
            self.ly_D = 3  # Number of layers for y
            self.ld_D = 5  # Number of layers after concatenation
            self.ux_D = 64  # Number of units in each layer for x
            self.uy_D = 64  # Number of units in each layer for y
            self.ud_D = 128 # Number of units in each layer after concatenation

            self.discriminator = self.build_discriminator(self.lx_D, self.ly_D, self.ld_D, self.ux_D, self.uy_D, self.ud_D)
            self.generator = self.build_generator()
        else:
            print("loading weights from local h5 weight")
            self.discriminator = tf.keras.models.load_model(self.discriminator_h5_path)
            self.generator = tf.keras.models.load_model(self.generator_h5_path)
        
        # Data Preparation 
        training_data_mat = scipy.io.loadmat(self.training_data_mat_path)
        x = training_data_mat['data']

        # Standardize the features
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(x)

        #Load Target Values
        target_data_mat = scipy.io.loadmat(self.target_data_mat_path)
        self.y = target_data_mat['data']

        #Load Test data
        testing_data_mat = scipy.io.loadmat(self.testing_data_mat_path)
        x_test = testing_data_mat['data']
        self.x_test = self.scaler.transform(x_test)

    def build_generator(self):
        input_layer = tf.keras.layers.Input(shape=(6,))  # TODO hot encoding 
        condition_layer = tf.keras.layers.Input(shape=(1,))
        noise_input = tf.keras.layers.Input(shape=(2,))

        # Initial concatenation
        merged_input = tf.keras.layers.concatenate([input_layer, condition_layer, noise_input])

        # First layer
        x = tf.keras.layers.Dense(128, activation='relu')(merged_input)

        # Subsequent layers with noise injection
        for _ in range(7):
            x = tf.keras.layers.concatenate([x, noise_input])  # Inject noise
            x = tf.keras.layers.Dense(128, activation='relu')(x)

        output = tf.keras.layers.Dense(1, activation='linear')(x)

        generator = tf.keras.Model(inputs=[input_layer, condition_layer, noise_input], outputs=output)

        return generator


    def build_discriminator(self,lx_D, ly_D, ld_D, ux_D, uy_D, ud_D):
        # Input layers
        x_input = tf.keras.layers.Input(shape=(1,))
        y_input = tf.keras.layers.Input(shape=(1,))

        # Processing x through lx;D layers of ux;D units
        x = x_input
        for _ in range(lx_D):
            x = tf.keras.layers.Dense(ux_D, activation='relu')(x)

        # Processing y through ly;D layers of uy;D units
        y = y_input
        for _ in range(ly_D):
            y = tf.keras.layers.Dense(uy_D, activation='relu')(y)

        # Concatenating the outputs
        concatenated = tf.keras.layers.concatenate([x, y])

        # Additional layers after concatenation
        z = concatenated
        for _ in range(ld_D):
            z = tf.keras.layers.Dense(ud_D, activation='relu')(z)

        # Final output layer
        output = tf.keras.layers.Dense(1, activation='sigmoid')(z)

        # Creating the model
        discriminator = tf.keras.Model(inputs=[x_input, y_input], outputs=output)
        return discriminator

        # Example usage with the specified layer structure
        discriminator = build_discriminator(lx_D=3, ly_D=3, ld_D=5, ux_D=256, uy_D=256, ud_D=256)


    def train(self, x_train, y_train, epochs=10):
        # Compile models
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate_discriminator), loss='binary_crossentropy')
        if self.USE_HUBER_LOSS_GENERATOR:
            # Choose a delta value for Huber loss
            delta_value = 1.0  # You can adjust this value based on the characteristics of your data

            # Compile the generator with Huber loss
            self.generator.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate_generator), loss=tf.keras.losses.Huber(delta=delta_value))
        else:
            self.generator.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate_generator), loss='mse')


        # Placeholder for labels
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))


        relative_errors_per_epoch = []
        start_time = time.time()

        with tqdm(range(epochs),mininterval=1) as pbar:
            for epoch in pbar:
                # Train discriminator
                idx = np.random.randint(0, self.x.shape[0], self.batch_size)
                real_x = self.x[idx]
                real_y = self.y[idx]

                condition = np.random.rand(self.batch_size, 1)  # create one-dimensional condition vector 
                noise = np.random.normal(0, 1, (self.batch_size, 2))

                # Pass all three inputs to the generator
                fake_y = self.generator.predict([real_x, condition, noise])


                real_y = np.reshape(real_y, (self.batch_size, 1))
                fake_y = np.reshape(fake_y, (self.batch_size, 1))

                d_loss_real = self.discriminator.train_on_batch([real_y, condition], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_y, condition], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # training generator
                g_loss = self.generator.train_on_batch([real_x, condition, noise], real_y)

                # Update the learning rate using the scheduler
                new_lr = lr_schedule(epoch)
                tf.keras.backend.set_value(self.generator.optimizer.lr, new_lr)
                tf.keras.backend.set_value(self.discriminator.optimizer.lr, new_lr*5)

                # calculate and record relative percentage error in this batch 
                relative_errors = np.mean(np.abs(real_y.flatten() - fake_y.flatten()) / np.abs(real_y.flatten())*100)
                relative_errors_per_epoch.append(np.mean(relative_errors))

                # print progress bar
                pbar.set_description(f'Average Relative Error: {np.mean(relative_errors):.2f}')
            end_time = time.time()
            total_time = end_time - start_time
            # Save the model
            self.generator.save(self.generator_h5_path)
            self.discriminator.save(self.discriminator_h5_path)


    def predict(self, x, USE_NOISY_CONDITION = False, USE_SCALER_TRANSFORMED=False):
        # Generate predictions
        if USE_NOISY_CONDITION:
            condition = np.random.rand(x.shape[0], 1)
            noise = np.random.rand(x.shape[0], 2)
        else:
            condition = np.ones((x.shape[0], 1))*0.5
            noise = np.ones((x.shape[0], 2))*0.5

        # for raw x data, need same scaler transform as the training data set
        if USE_SCALER_TRANSFORMED :
            x = self.scaler.transform(x)
        else:
            pass
        predictions = self.generator.predict([x, condition, noise])

        return predictions

    def save_model(self):
        self.generator.save(self.generator_h5_path)
        self.discriminator.save(self.discriminator_h5_path)

    def load_model(self, filepath):
        self.discriminator = tf.keras.models.load_model(self.generator_h5_path)
        self.generator = tf.keras.models.load_model(self.discriminator_h5_path)

class MagNetDatasetPreparation():
    # simply run datasetPreparation(), 
    def __init__(self):
        pass
    def identify_waveform_type(self, waveform, frequency):
        # Check if the waveform is a sine wave
        second_derivative = np.diff(np.diff(waveform))
        waveform_truncated = waveform[2:]
        correlation_matrix = np.corrcoef(-waveform_truncated, second_derivative)

        if correlation_matrix[0, 1] > 0.8:
            return 'Sine'

        # If not a sine wave, check if it's triangular or trapezoidal
        # Calculate the sampling interval (time between samples)
        sampling_interval = 1 / frequency

        # Create a time vector corresponding to the waveform
        time = np.arange(len(waveform)) * sampling_interval

        # Find indices where B-value is positive
        positive_indices = np.where(waveform > 0)[0]

        # Split the indices into continuous segments
        segments = np.where(np.diff(positive_indices) > 1)[0]
        start_idx = np.insert(segments + 1, 0, 0)
        end_idx = np.append(segments, len(positive_indices) - 1)

        # Initialize variables for area calculations
        total_area = 0
        total_max_product = 0

        # Iterate through each segment
        for i in range(len(start_idx)):
            # Extract segment
            segment_indices = positive_indices[start_idx[i]:end_idx[i] + 1]
            segment_waveform = waveform[segment_indices]
            segment_time = time[segment_indices]

            # Calculate area under the curve for this segment
            segment_area = np.trapz(segment_waveform, segment_time)
            total_area += segment_area

            # Calculate the product of maximum value and time length
            max_b_value = np.max(segment_waveform)
            time_length = segment_time[-1] - segment_time[0]
            max_product = max_b_value * time_length
            total_max_product += max_product

        # Determine if the waveform is triangular or trapezoidal
        if abs(total_area - total_max_product * 0.5) < total_area * 0.05:
            return 'Triangular'
        else:
            return 'Trapezoidal'

    def trainingDatasetPreparation(self,material_type="A", training_dataset_path='..\\dataset\\train',interim_path=".\\data\\interim"):
        # Output: Generate a intermediate .mat file containing processed data for cGAN model input.
        # Validate the input
        valid_materials = {'A', 'B', 'C', 'D', 'E'}
        if material_type not in valid_materials:
            raise ValueError('Invalid material. Please enter A, B, C, D, or E.')

        # Format the folder name based on the chosen material
        folder_path = os.path.join(training_dataset_path, f'Material {material_type}')  # Adjust the path format if necessary
        extension = '*.csv'

        # Open the folder in the default file explorer
        os.startfile(folder_path)

        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, extension))
        required_files = {'B_Field.csv', 'Frequency.csv', 'Temperature.csv', 'Volumetric_Loss.csv'}
        found_files = set(os.path.basename(filename) for filename in csv_files)

        # Check if all required files are present
        if not required_files.issubset(found_files):
            raise FileNotFoundError(f"Missing required files in {folder_path}")

        # Read files for the chosen material
        b_field = pd.read_csv(os.path.join(folder_path, 'B_Field.csv'), header=None)
        frequency = pd.read_csv(os.path.join(folder_path, 'Frequency.csv'), header=None)
                
        # Initialize the arrays for each waveform type
        num_waveforms = len(b_field)
        sinew = np.zeros(num_waveforms)
        triw = np.zeros(num_waveforms)
        trapw = np.zeros(num_waveforms)

        for i in range(num_waveforms):
            waveform = b_field.iloc[i, :].values
            freq = frequency.iloc[i, :].values

            wavetype = self.identify_waveform_type(waveform, freq)

            if wavetype == 'Sine':
                sinew[i] = 1
            elif wavetype == 'Triangular':
                triw[i] = 1
            elif wavetype == 'Trapezoidal':
                trapw[i] = 1

        waveform_shape = np.column_stack([sinew, triw, trapw])

        # Calculate the maximum absolute flux density
        max_abs_flux = b_field.abs().max(axis=1).values

        # Convert Frequency DataFrame to array
        frequency_array = frequency.values


        # Read files for the chosen material
        temperature = pd.read_csv(os.path.join(folder_path, 'Temperature.csv'), header=None)
        volumetric_loss = pd.read_csv(os.path.join(folder_path, 'Volumetric_Loss.csv'), header=None)

        # Concatenate the arrays
        training = np.column_stack([waveform_shape, max_abs_flux, frequency_array, temperature])

        # Save the merged data into a .mat file
        scipy.io.savemat(os.path.join(interim_path,f'{material_type}_training.mat'), {'data': training})
        scipy.io.savemat(os.path.join(interim_path,f'{material_type}_measurements.mat'), {'data': volumetric_loss})

    def testingDatasetPreparation(self,material_type="A",testing_dataset_path='..\\dataset\\test',interim_path=".\\data\\interim"):
        # Validate the input
        valid_materials = {'A', 'B', 'C', 'D', 'E'}
        if material_type not in valid_materials:
            raise ValueError('Invalid material. Please enter A, B, C, D, or E.')

        # Format the folder name based on the chosen material
        folder_path = os.path.join(testing_dataset_path, f'Material {material_type}')  # Adjust the path format if necessary
        extension = '*.csv'

        # Open the folder in the default file explorer
        os.startfile(folder_path)

        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, extension))
        required_files = {'B_Field.csv', 'Frequency.csv', 'Temperature.csv'}
        found_files = set(os.path.basename(filename) for filename in csv_files)

        # Check if all required files are present
        if not required_files.issubset(found_files):
            raise FileNotFoundError(f"Missing required files in {folder_path}")

        # Read files for the chosen material
        b_field = pd.read_csv(os.path.join(folder_path, 'B_Field.csv'), header=None)
        frequency = pd.read_csv(os.path.join(folder_path, 'Frequency.csv'), header=None)
        
        # Initialize the arrays for each waveform type
        num_waveforms = len(b_field)
        sinew = np.zeros(num_waveforms)
        triw = np.zeros(num_waveforms)
        trapw = np.zeros(num_waveforms)

        for i in range(num_waveforms):
            waveform = b_field.iloc[i, :].values
            freq = frequency.iloc[i, :].values

            wavetype = self.identify_waveform_type(waveform, freq)

            if wavetype == 'Sine':
                sinew[i] = 1
            elif wavetype == 'Triangular':
                triw[i] = 1
            elif wavetype == 'Trapezoidal':
                trapw[i] = 1

        waveform_shape = np.column_stack([sinew, triw, trapw])

        # Calculate the maximum absolute flux density
        max_abs_flux = b_field.abs().max(axis=1).values

        # Convert Frequency DataFrame to array
        frequency_array = frequency.values

        # Read files for the chosen material
        temperature = pd.read_csv(os.path.join(folder_path, 'Temperature.csv'), header=None)

        # Concatenate the arrays
        test = np.column_stack([waveform_shape, max_abs_flux, frequency_array, temperature])

        # Save the merged data into a .mat file
        scipy.io.savemat(os.path.join(interim_path,f'{material_type}_testing.mat'), {'data': test})
        




if __name__ == "__main__":
    
    
    model = cGAN()
