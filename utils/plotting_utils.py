import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import csv

def plot_gradients(datafile):
    """
    Plot 3-d representation of gradient 
    """
    # Open data file and read out gradient for 
    # given epoch
    with open(datafile, 'r') as f:
        csv_file = csv.reader(f, delimiter=',')

        gradients = []
        for line in csv_file:
            if line:
                gradients.append([float(val) for val in line])

    gradients = gradients[1:]
    gradients = gradients[:10]
    num_time_steps = np.arange(start=len(gradients), stop=0, step=-1)
    num_units = np.arange(len(gradients[0]))
    X, Y = np.meshgrid(num_units, num_time_steps)
    gradients = np.array(gradients)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, gradients,
            cmap='viridis', edgecolor='green')
    ax.set_xlabel("Reccurrent Unit No")
    ax.set_ylabel("Time Step")
#    ax.set_ylim([len(gradients)-10, len(gradients)])
#    ax.set_zlim([-1e-3, 1e-3])
    plt.show()

def plot_validation_accuracy(datafile):
    """
    Plot validation accuracy over epochs
    """
    with open(datafile, 'r') as f:
        lines = f.readlines()

    accuracies = np.array([float(val) for val in lines])

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(accuracies)
    ax.set_title("Validation Accuracy over Epoch")
    ax.set_xlabel("Epoch No")
    ax.set_ylabel("Accuracy")
    plt.show()

def plot_loss(datafile,
              run="Training"):
    """
    Plot MSE error over epochs
    """
    with open(datafile, 'r') as f:
        lines = f.readlines()

    errors = np.array([float(val) for val in lines])

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(errors)
    ax.set_title(run + " Error over Epoch")
    ax.set_xlabel("Epoch No")
    ax.set_ylabel("Error")
    plt.show()

def plot_vowel_data(datafile, pid="Speaker 1"):
    """
    Plot 3-d graph of vowel data 
    """
    with open(datafile, 'r') as f:
        csv_file = csv.reader(f, delimiter=' ')
        data_block = []
        for line in csv_file:
            data_block.append([float(val) for val in line[:12]])

    num_time_steps = np.arange(len(data_block))
    num_units = np.arange(len(data_block[0]))
    X, Y = np.meshgrid(num_units, num_time_steps)
    data = np.array(data_block)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, data,
            cmap='plasma', edgecolor='purple')
    ax.set_title("Plot of LPC Cepstrum Coefficients for {}".format(id))
    ax.set_ylabel("Time Step")
    ax.set_xlabel("LPC Cepstrum Coefficient Number")
    plt.show()

def plot_sign_data(datafile, sign, pid="Andrew"):
    """
    Plot 3-d graph of sign data 
    """
    with open(datafile, 'r') as f:
        csv_file = csv.reader(f, delimiter=',')
        data_block = []
        for line in csv_file:
            data_block.append([float(val) for val in line[:11]])

    num_time_steps = np.arange(len(data_block))
    num_units = np.arange(len(data_block[0]))
    X, Y = np.meshgrid(num_units, num_time_steps)
    data = np.array(data_block)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, data,
                cmap=cm.coolwarm)
    ax.set_title("Readings from %s: %s" % (pid, sign))
    ax.set_ylabel("Time Step")
    ax.set_xlabel("PowerGlove Reading")
    plt.show()

def plot_error_chart(train_datafile, 
                     val_datafile,
                     title="Loss over Epoch"):

    training_errors = []
    with open(train_datafile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            training_errors.append(float(l))

    print("training_errors: ", training_errors)

    validation_errors = []
    with open(val_datafile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            validation_errors.append(float(l))

    print("validation_errors: ", validation_errors)

    training_erros = np.array(training_errors)
    validation_errors = np.array(validation_errors)

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(training_errors, label='Training Loss')
    ax.plot(validation_errors, label='Validation Loss') 
    plt.legend(loc='best')

    plt.show()



if __name__ == '__main__':
    trainfile = "logs/MitchellRNNv2_train_error_trainSigns.txt"
    valfile = "logs/MitchellRNNv2_val_error_trainSigns.txt"
    plot_error_chart(trainfile, valfile)



