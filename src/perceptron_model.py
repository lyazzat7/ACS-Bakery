import joblib
import pandas as pd
from numpy import arange, zeros
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report
from util.value_generator import ValueGenerator


class PerceptronModel:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                 learning_rate='constant', max_iter=200, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def create_dataset_for_classifier_model(self, size, min_temperature, max_temperature, min_humidity, max_humidity,
                                            ref_temp, ref_humid):
        """
        Generates a dataset where temperature and humidity gradually approach their respective setpoints.

        Args:
        - size (int): Number of data points to generate.
        - initial_temp (float): Starting temperature.
        - initial_humid (float): Starting humidity.
        - ref_temp (float): Reference or setpoint temperature.
        - ref_humid (float): Reference or setpoint humidity.
        """
        temperatures = ValueGenerator.generate_uniform_data(min_temperature, max_temperature, min_temperature, size)
        humidity = ValueGenerator.generate_uniform_data(min_humidity, max_humidity, min_humidity, size)

        temp_diff = temperatures - ref_temp
        humid_diff = humidity - ref_humid

        actions = []
        for t_diff, h_diff in zip(temp_diff, humid_diff):
            if t_diff > 0:
                action_temp = 0  # Reduce temperature
            elif t_diff < 0:
                action_temp = 1  # Increase temperature
            else:
                action_temp = 2  # Hold temperature

            if h_diff > 0:
                action_humid = 0  # Reduce humidity
            elif h_diff < 0:
                action_humid = 1  # Increase humidity
            else:
                action_humid = 2  # Hold humidity

            actions.append((action_temp, action_humid))

        df = pd.DataFrame({
            'Temperature': temperatures,
            'Humidity': humidity,
            'Temp Action': [a[0] for a in actions],
            'Humid Action': [a[1] for a in actions]
        })
        return df

    def create_dataset_for_regressor_model(self, time, initial_temp, initial_humid, ref_temp, ref_humid, Kp, Ki, Kd,
                                           time_step=1):
        """
        Generates a dataset for regression simulating PID controller adjustments towards setpoints for temperature and humidity.
        Adds time column representing the time in seconds at each step.

        Args:
        - time (int): Time in ms.
        - initial_temp (float): Initial temperature value.
        - initial_humid (float): Initial humidity value.
        - ref_temp (float): Reference or setpoint temperature.
        - ref_humid (float): Reference or setpoint humidity.
        - Kp (float): Proportional gain for PID.
        - Ki (float): Integral gain for PID.
        - Kd (float): Derivative gain for PID.
        - time_step (int or float): Time interval between PID adjustments, default is 1 second.
        """
        # Initialize arrays
        temperatures = zeros(time)
        humidity = zeros(time)
        temperatures[0] = initial_temp
        humidity[0] = initial_humid

        # Errors and derivatives
        temp_errors = zeros(time)
        humid_errors = zeros(time)
        temp_integral = 0
        humid_integral = 0

        # Time array
        time_arr = arange(0, time * time_step, time_step)

        # Simulate PID adjustments
        for i in range(1, time):
            # Calculate errors
            temp_errors[i] = ref_temp - temperatures[i - 1]
            humid_errors[i] = ref_humid - humidity[i - 1]

            # Update integral
            temp_integral += temp_errors[i]
            humid_integral += humid_errors[i]

            # Derivative
            temp_derivative = temp_errors[i] - temp_errors[i - 1]
            humid_derivative = humid_errors[i] - humid_errors[i - 1]

            # PID output
            temp_change = Kp * temp_errors[i] + Ki * temp_integral + Kd * temp_derivative
            humid_change = Kp * humid_errors[i] + Ki * humid_integral + Kd * humid_derivative

            # Update values
            temperatures[i] = temperatures[i - 1] + temp_change
            humidity[i] = humidity[i - 1] + humid_change

        # Create DataFrame
        df = pd.DataFrame({
            'Time': time_arr,
            'Temperature': temperatures,
            'Humidity': humidity,
            'Temperature Error': temp_errors,
            'Humidity Error': humid_errors
        })
        return df

    def create_dataset_for_regressor_multistage_model(self, total_time,
                                                      stage_params, Kp, Ki, Kd, time_step=1,
                                                      transition_steps=10):
        temperatures = zeros(total_time)
        humidity = zeros(total_time)
        temp_errors = zeros(total_time)
        humid_errors = zeros(total_time)

        stage_durations = [total_time // 3, total_time // 3, total_time - 2 * (total_time // 3)]
        stage_end_times = [sum(stage_durations[:i + 1]) for i in range(len(stage_durations))]

        temperatures[0] = stage_params[0]['initial_temp']
        humidity[0] = stage_params[0]['initial_humid']

        time_arr = arange(0, total_time * time_step, time_step)
        current_stage = 0
        next_transition_index = stage_end_times[current_stage] - transition_steps

        temp_integral = 0
        humid_integral = 0

        for i in range(1, total_time):
            if i >= next_transition_index and current_stage < len(stage_params) - 1:
                transition_fraction = (i - next_transition_index) / float(transition_steps)
                ref_temp = stage_params[current_stage]['ref_temp'] * (1 - transition_fraction) + \
                           stage_params[current_stage + 1]['ref_temp'] * transition_fraction
                ref_humid = stage_params[current_stage]['ref_humid'] * (1 - transition_fraction) + \
                            stage_params[current_stage + 1]['ref_humid'] * transition_fraction
            else:
                ref_temp = stage_params[current_stage]['ref_temp']
                ref_humid = stage_params[current_stage]['ref_humid']

            if i == stage_end_times[current_stage]:
                if current_stage < len(stage_params) - 1:
                    current_stage += 1
                    next_transition_index = stage_end_times[current_stage] - transition_steps
                temp_integral = 0
                humid_integral = 0

            temp_errors[i] = ref_temp - temperatures[i - 1]
            humid_errors[i] = ref_humid - humidity[i - 1]

            temp_integral += temp_errors[i]
            humid_integral += humid_errors[i]

            temp_derivative = temp_errors[i] - temp_errors[i - 1]
            humid_derivative = humid_errors[i] - humid_errors[i - 1]

            temp_change = Kp * temp_errors[i] + Ki * temp_integral + Kd * temp_derivative
            humid_change = Kp * humid_errors[i] + Ki * humid_integral + Kd * humid_derivative

            temperatures[i] = temperatures[i - 1] + temp_change
            humidity[i] = humidity[i - 1] + humid_change

        df = pd.DataFrame({
            'Time': time_arr,
            'Temperature': temperatures,
            'Humidity': humidity,
            'Temperature Error': temp_errors,
            'Humidity Error': humid_errors
        })
        return df

    def train_test_split_for_classifier_model(self, df):
        X = df[['Temperature', 'Humidity']]
        y = pd.DataFrame({'Temp Action': df['Temp Action'], 'Humid Action': df['Humid Action']})
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def train_test_split_for_regressor_model(self, df):
        X = df[['Temperature', 'Humidity']]
        y = df[['Temperature Error', 'Humidity Error']]
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def preprocess_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_classifier_model(self, X_train, y_train):
        mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                            solver=self.solver, alpha=self.alpha, learning_rate=self.learning_rate,
                            max_iter=self.max_iter, random_state=self.random_state)
        multi_output_mlp = MultiOutputClassifier(mlp)
        multi_output_mlp.fit(X_train, y_train)
        return multi_output_mlp

    def train_regressor_model(self, X_train, y_train):
        mlp = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                           solver=self.solver, alpha=self.alpha, learning_rate=self.learning_rate,
                           max_iter=self.max_iter, random_state=self.random_state)
        multi_output_mlp = MultiOutputRegressor(mlp)
        multi_output_mlp.fit(X_train, y_train)
        return multi_output_mlp

    def evaluate_classifier_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        return classification_report(y_test, predictions, zero_division=0)

    def evaluate_regressor_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, mae, r2

    def train_or_load_classifier_model(self, X_train, y_train, filename='model.pkl'):
        model = load_model(filename)
        if model is None:
            self.train_classifier_model(X_train, y_train)
        save_model(model, filename)
        return model

    def train_or_load_regressor_model(self, X_train, y_train, filename='model.pkl'):
        model = load_model(filename)
        if model is None:
            self.train_regressor_model(X_train, y_train)
        save_model(model, filename)
        return model


def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)


def load_model(filename='model.pkl'):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        return None
