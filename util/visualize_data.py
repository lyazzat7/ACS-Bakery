from turtledemo.chaos import plot

import matplotlib.pyplot as plt
import numpy as np


def build_graphs_for_classifier_model(df, ref_temp, ref_humid,
                                      filename_to_save='classifier_model_plots.png',
                                      filename_to_save_actions='classifier_model_actions_plots.png'):
    fig = plt.figure(figsize=(14, 6))

    # Temperature plot
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df['Temperature'], label='Actual Temperature')
    plt.plot(df.index, [ref_temp] * len(df), 'r--', label='Reference Temperature')
    plt.title('Temperature Over Number of Measurements')
    plt.xlabel('Number of Measurements, count')
    plt.ylabel("Temperature, °C")
    plt.legend()

    # Humidity plot
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df['Humidity'], label='Actual Humidity')
    plt.plot(df.index, [ref_humid] * len(df), 'r--', label='Reference Humidity')
    plt.title('Humidity Over Number of Measurements')
    plt.xlabel('Number of Measurements, count')
    plt.ylabel('Humidity, %')
    plt.legend()

    fig_actions = plt.figure(figsize=(14, 6))

    # Temperature Actions plot
    plt.subplot(1, 2, 1)
    plt.step(df.index, df['Temp Action'], where='mid', label='Temperature Action', color='g')
    plt.title('Temperature Actions Over Measurements')
    plt.xlabel('Number of Measurements, count')
    plt.ylabel('Action (0=Decrease, 1=Increase, 2=Hold)')
    plt.yticks([0, 1, 2], ['Decrease', 'Increase', 'Hold'])
    plt.legend()

    # Humidity Actions plot
    plt.subplot(1, 2, 2)
    plt.step(df.index, df['Humid Action'], where='mid', label='Humidity Action', color='b')
    plt.title('Humidity Actions Over Measurements')
    plt.xlabel('Number of Measurements, count')
    plt.ylabel('Action (0=Decrease, 1=Increase, 2=Hold)')
    plt.yticks([0, 1, 2], ['Decrease', 'Increase', 'Hold'])
    plt.legend()

    plt.tight_layout()
    fig.savefig(filename_to_save)
    fig_actions.savefig(filename_to_save_actions)
    plt.show()


def build_graphs_for_regressor_model(df, ref_temp, ref_humid, cur_temp, cur_humid,
                                     filename_to_save='regressor_model_plots.png'):
    time = df['Time'].to_numpy()

    # Calculate dynamic characteristics for Temperature
    temp_rise_time = calculate_rise_time(df['Temperature'], time, ref_temp)
    temp_overshoot = calculate_overshoot(df['Temperature'], ref_temp)
    temp_settling_time = calculate_settling_time(df['Temperature'].to_numpy(), time, ref_temp)

    # Calculate dynamic characteristics for Humidity
    humid_rise_time = calculate_rise_time(df['Humidity'], df['Time'], ref_humid)
    humid_overshoot = calculate_overshoot(df['Humidity'], ref_humid)
    humid_settling_time = calculate_settling_time(df['Humidity'].to_numpy(), time, ref_humid)

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['Time'], df['Temperature'], label='Actual Temperature')
    plt.axhline(y=ref_temp, color='r', linestyle=':', label='Setpoint Temperature')
    plt.axhline(y=cur_temp, color='g', linestyle='dashdot', label='Initial Temperature')
    plt.xlabel("Time, ms")
    plt.ylabel("Temperature, °C")
    plt.title("Temperature Control Over Time")
    # plt.title(
    #     f'Temperature Control Over Time\nRise Time: {temp_rise_time}ms, Overshoot: {temp_overshoot}%, Settling Time: {temp_settling_time}ms')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df['Time'], df['Humidity'], label='Actual Humidity')
    plt.axhline(y=ref_humid, color='r', linestyle=':', label='Setpoint Humidity')
    plt.axhline(y=cur_humid, color='g', linestyle='dashdot', label='Initial Humidity')
    plt.xlabel("Time, ms")
    plt.ylabel("Humidity, %")
    plt.title("Humidity Control Over Time")
    # plt.title(
    #     f'Humidity Control Over Time\nRise Time: {humid_rise_time}ms, Overshoot: {humid_overshoot}%, Settling Time: {humid_settling_time}ms')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig.savefig(filename_to_save)
    plt.show()


def build_graphs_for_regressor_multistage_model(df, stage_params,
                                                filename_to_save='multi_stage_regressor_model_plots.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    time_splits = np.cumsum([0] + [params['duration'] for params in stage_params])

    colors = ['blue', 'green', 'red']
    labels = ['Stage 1', 'Stage 2', 'Stage 3']

    ax1.set_title('Temperature Control Over Time')
    ax1.set_xlabel('Time, ms')
    ax1.set_ylabel('Temperature, °C')

    for i, (params, color, label) in enumerate(zip(stage_params, colors, labels)):
        start_idx = time_splits[i]
        end_idx = time_splits[i + 1]
        ax1.plot(df['Time'][start_idx:end_idx], df['Temperature'][start_idx:end_idx], color=color,
                 label=f'{label} (Target: {params["ref_temp"]}°C)')
        ax1.axvline(df['Time'][start_idx], color='grey', linestyle='--')
    ax1.axhline(y=stage_params[-1]['ref_temp'], color='r', linestyle=':', label='Final Setpoint Temperature')

    temperature = [136, 121, 59.8, 91.4, 118, 102, 81, 25.9, 114, 233, 265, 222, 247, 57.5, 270, 275, 245, 224, 177,
                   109, 170, 89.1, 145, 202, 155, 258, 98.1]
    time_stage1 = np.linspace(df['Time'].iloc[0], df['Time'].iloc[time_splits[1] - 1], len(temperature[:9]))
    time_stage2 = np.linspace(df['Time'].iloc[time_splits[1]], df['Time'].iloc[time_splits[2] - 1],
                              len(temperature[9:18]))
    time_stage3 = np.linspace(df['Time'].iloc[time_splits[2]], df['Time'].iloc[-1], len(temperature[18:]))

    ax1.plot(time_stage1, temperature[:9], color='blue', linestyle='--', label='Real Experiment Stage 1')
    ax1.plot(time_stage2, temperature[9:18], color='green', linestyle='--', label='Real Experiment Stage 2')
    ax1.plot(time_stage3, temperature[18:], color='red', linestyle='--', label='Real Experiment Stage 3')

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim([df['Time'][0], df['Time'].iloc[-1]])

    ax2.set_title('Humidity Control Over Time')
    ax2.set_xlabel('Time, ms')
    ax2.set_ylabel('Humidity, %')
    for i, (params, color, label) in enumerate(zip(stage_params, colors, labels)):
        start_idx = time_splits[i]
        end_idx = time_splits[i + 1]
        ax2.plot(df['Time'][start_idx:end_idx], df['Humidity'][start_idx:end_idx], color=color,
                 label=f'{label} (Target: {params["ref_humid"]}%)')
        ax2.axvline(df['Time'][start_idx], color='grey', linestyle='--')
    ax2.axhline(y=stage_params[-1]['ref_humid'], color='r', linestyle=':', label='Final Setpoint Humidity')

    humidity = [81.3, 71.3, 38.7, 22, 35.5, 37, 62, 79.6, 39.6, 16.8, 23.2, 32, 47.3, 16.8, 75.8, 23, 24.4, 30.8, 18.1,
                37.1, 16.8, 59.9, 21.9, 14.3, 55.5, 33, 75.8]
    ax2.plot(time_stage1, humidity[:9], color='blue', linestyle='--', label='Real Experiment Stage 1')
    ax2.plot(time_stage2, humidity[9:18], color='green', linestyle='--', label='Real Experiment Stage 2')
    ax2.plot(time_stage3, humidity[18:], color='red', linestyle='--', label='Real Experiment Stage 3')

    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim([df['Time'][0], df['Time'].iloc[-1]])

    plt.tight_layout()
    fig.savefig(filename_to_save)
    plt.show()


def calculate_rise_time(y, time, ref, rise_percents=(0.1, 0.9)):
    start_value = ref * rise_percents[0]
    end_value = ref * rise_percents[1]

    start_time = None
    end_time = None

    for i, value in enumerate(y):
        if value >= start_value and start_time is None:
            start_time = time[i]
        if value >= end_value and start_time is not None:
            end_time = time[i]
            break

    if start_time is not None and end_time is not None:
        return end_time - start_time
    else:
        return None


def calculate_overshoot(y, ref):
    return max(0, max(y) - ref)


def calculate_settling_time(y, time, ref, tolerance=0.05):
    lower_bound = ref * (1 - tolerance)
    upper_bound = ref * (1 + tolerance)
    settled = False

    for i, value in enumerate(y):
        if lower_bound <= value <= upper_bound:
            if not settled:
                settling_start_time = time[i]
                settled = True
        else:
            settled = False

    if settled:
        return settling_start_time
    return None
