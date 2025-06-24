import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def build_array(length, entry_points, fill_method="ffil"):
    """
    Build an array of specified length using the provided entry points.

    Args:
        length (int): Desired length of the output array
        entry_points (list): List of tuples (x, t) where x is a value and t is an index
        fill_method (str): Method to fill missing values. Options are 'zero' or 'ffill' (forward fill)

    Returns:
        list: An array of the specified length with values at entry points and filled values elsewhere
    """
    # Initialize array with zeros
    result = [0] * length

    # Sort entry points by index
    sorted_entries = sorted(entry_points, key=lambda pair: pair[1])

    # Process entry points
    for x, t in sorted_entries:
        if 0 <= t < length:  # Check if index is within bounds
            result[t] = x

    # If using forward fill
    if fill_method == "ffill":
        last_valid_value = None
        for i in range(length):
            if result[i] != 0:
                last_valid_value = result[i]
            elif last_valid_value is not None:
                result[i] = last_valid_value

    return result


def plot_chronogram(
    timesteps: list[float],
    voltage_log: dict[str, list[tuple]],
    spike_log: dict[str, list[float]],
):
    print("Launching chronogram visualization...")
    print("=========================================")
    n = len(voltage_log.keys())
    _, ax = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(10, 5))
    values = [
        i / (n - 1) for i in range(n)
    ]  # linearly spaced values between 0 and 1
    colors = iter(cm.rainbow(values))

    for i, item in enumerate(voltage_log.keys()):
        c = next(colors)
        v_log = voltage_log[item]
        if len(timesteps) != len(v_log):
            v_log = build_array(len(timesteps), v_log)
        ax[i].plot(timesteps, v_log, c=c)
        # Neuron names
        ax[i].set_ylabel(item, rotation=0, labelpad=30)
        # Voltage limits
        ax[i].set_ylim(-15, 15)
        # Removing extra graphical elements (axes ticks and graph spines)
        ax[i].get_yaxis().set_ticks([])
        ax[i].xaxis.set_tick_params(length=0)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)

        if item in spike_log:
            for spike in spike_log[item]:
                ax[i].scatter(spike, 0, s=20, c=c)

    plt.show()
    print("=========================================")


if __name__ == "__main__":
    import numpy as np

    def build_array_with_padding(desired_length, valid_entry_points):
        # Initialize an array of zeros with the desired length
        result_array = np.zeros(desired_length)

        # Convert the list of valid entry points to a numpy array for efficient indexing
        x_values = np.array([x for x, _ in valid_entry_points])
        t_indices = np.array([t for _, t in valid_entry_points])

        # Find the last valid x value before each timestep
        cummax_x = np.maximum.accumulate(x_values)
        cummax_t = np.maximum.accumulate(t_indices)

        # Create a mask for where valid entries are present
        valid_mask = np.zeros(desired_length, dtype=bool)
        valid_mask[t_indices] = True

        # Fill the array with the last valid x value encountered up to each timestep
        result_array[: cummax_t[-1] + 1] = cummax_x[-1]
        np.maximum.at(result_array, t_indices, x_values)

        return result_array

    # Example usage:
    desired_length = 10
    valid_entry_points = [(2, 1), (3, 2), (4, 3)]
    result = build_array_with_padding(desired_length, valid_entry_points)
    print(result)  # Output: [0.  0.  0.  2.  3.  4.  4.  4.  4.  4.]
