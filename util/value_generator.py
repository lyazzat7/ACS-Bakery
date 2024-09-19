import numpy as np
from simple_pid import PID


class ValueGenerator:
    def __init__(self, seed=None):
        """
        Initializes the ValueGenerator with an optional seed for reproducibility.

        Args:
        - seed (int, optional): Seed for the random number generator. Default is None.
        """
        if seed is not None:
            np.random.seed(seed)
        else:
            seed = 0

    @staticmethod
    def generate_uniform_data(lower_bound: float, upper_bound: float, initial_value: float, count: int,
                              step: float = 0.1):
        """
        Generates temperature data starting from an initial value with small random steps,
        constrained by minimum and maximum temperatures.

        Args:
        - lower_bound (float): Minimum allowable value.
        - upper_bound (float): Maximum allowable value.
        - initial_value (float): Initial value.
        - count (int): The number of samples to generate.
        - step (float): Maximum change between consecutive values.

        Returns:
        - np.ndarray: An array of temperature values changing gradually within the specified limits.
        """
        vals = [initial_value]
        for _ in range(1, count):
            change = np.random.uniform(-step, step)  # Generate a small random change
            new_temp = vals[-1] + change  # Apply the change to the last temperature

            new_temp = max(lower_bound, new_temp)
            new_temp = min(upper_bound, new_temp)
            vals.append(new_temp)
        return np.array(vals)

    @staticmethod
    def gradual_approach(lower_bound: float, upper_bound: float, initial_value: float, target_value: float,
                         count: int, step: float = 0.1):
        """
        Generates data that gradually approaches a target value from an initial value,
        maintaining the values within specified bounds.

        Args:
        - initial_value (float): Initial value.
        - target_value (float): Target value.
        - count (int): Number of steps to generate.
        - step (float): Maximum change per step.
        - lower_bound (float, optional): Minimum allowable value.
        - upper_bound (float, optional): Maximum allowable value.

        Returns:
        - np.ndarray: An array of values gradually approaching the target within the bounds.
        """
        values = [initial_value]
        for _ in range(1, count):
            # Calculate change needed
            change_needed = target_value - values[-1]

            # Determine the actual change, ensuring it doesn't exceed the step size
            change = np.clip(change_needed, -step, step)

            # Apply the change and ensure the new value is within the specified bounds
            new_value = values[-1] + change
            if lower_bound is not None:
                new_value = max(lower_bound, new_value)
            if upper_bound is not None:
                new_value = min(upper_bound, new_value)

            values.append(new_value)
        return np.array(values)
