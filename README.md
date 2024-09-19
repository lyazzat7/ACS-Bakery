# Perceptron Model Training and Visualization Tool

## Overview

This Python project provides tools for creating, training, 
and evaluating machine learning models 
(specifically perceptron models) for temperature 
and humidity control. The models can be used for 
classification and regression tasks, with support for 
multi-output classification, regression, and multi-stage 
regression models. It also includes a GUI for model 
configuration and visualization, leveraging `PyQt5`.

## Project Structure

- **perceptron_model.py**: Contains the `PerceptronModel` class that handles dataset generation, training, testing, and evaluation of perceptron models. Supports both classifier and regressor models, with additional functionality for multi-stage models.
- **configuration_window.py**: Implements the `ConfigurationWindow` class, which provides a graphical user interface (GUI) for configuring model parameters, training the model, and visualizing results.
- **results_window.py**: Provides a window for displaying model performance metrics, particularly classification results.
- **value_generator.py**: Contains the `ValueGenerator` class, which helps generate simulated data (temperature and humidity) for training the perceptron models.
- **visualize_data.py**: Includes utilities for plotting the results of trained models, such as temperature and humidity over time, along with their respective actions.
- **util.py**: A utility script that contains helper functions for validating input values.
- **dependecy_installer.py**: Handles automatic dependency installation for required Python packages.

## Dependencies

The project uses several Python libraries, which can be installed using the provided dependecy_installer.py script. The key dependencies include:

- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `sklearn`: For building and evaluating machine learning models.
- `PyQt5`: For the graphical user interface (GUI).
- `joblib`: For model saving and loading.

## Graphical Interface

The application provides an easy-to-use GUI for configuring and training models. Key features include:

- **Model Parameters Configuration**: Set parameters like dataset size, temperature/humidity limits, activation function, solver, and more.
- **Training Options**: Train models for classification, regression, or multi-stage regression tasks.
- **Results Visualization**: View the results through interactive graphs, including temperature and humidity trends, and classification action decisions.
