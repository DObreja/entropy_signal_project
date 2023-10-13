import itertools
import time
import numpy as np

from numba import njit

# Below function will return an np array and corresponding time.
def rve_of_singal(time_signal, signal, frequency, τ_const = 1, epsilon_step = 4, samples_per_window = 5):

    clocker_start = time.time()

    # Creates a window that gets an amount of variables AKA: [1, 2, 3, 4] -> [1, 2], [2, 3],...
    total_bins = np.math.factorial(samples_per_window)

    # This will get us all the ranked values needed for the window.
    signal_noised_window = np.lib.stride_tricks.sliding_window_view(signal,
                                                                    samples_per_window * epsilon_step - (
                                                                                epsilon_step - 1))
    signal_noised_window = signal_noised_window[:, ::epsilon_step]  # This will step over 4 values.
    signal_noised_window_indices = np.argsort(signal_noised_window, axis=1)

    # Get all of the "permutations" that can be done with the provided samples per window. So it should be the factorial!
    all_to_bin_list_sequential = np.array(list(itertools.permutations(range(samples_per_window), samples_per_window)))

    # Generates the array where we will bin the values.
    histy_stuff = np.zeros(total_bins)
    histy_stuff[:] = 1  # We need to set all values to 1 initially.

    # For the window, how big should the values be?
    # This is important as we would need to dampen the values based on an over time function:
    # α = exp(−1/fτ) for the over time function.
    alpha_filter = np.exp(-1 / (frequency * τ_const))

    # Indeed next we will need to take the while loop so we only take x amount of variables into account. Make it >30
    max_section_width = np.size(signal_noised_window_indices, axis=0)

    # And so we get our histogram from this.
    entropy_arrays = optimised_rve(signal_noised_window_indices, histy_stuff, alpha_filter, all_to_bin_list_sequential, total_bins, max_section_width, samples_per_window)

    clocker_for_loop = time.time()
    print(clocker_for_loop - clocker_start)

    time_entropy = time_signal[:np.size(entropy_arrays, axis=0)]

    return entropy_arrays, time_entropy

# Realistically we want to average the outputted entropy first. Below effectively smoothens curve.
def entropy_window_averager(time_entropy, entropy_arrays):

    size_of_window = 400 # Window averaging size.
    to_average_entropy = np.lib.stride_tricks.sliding_window_view(entropy_arrays, size_of_window)
    entropy_averaged = np.sum(to_average_entropy, axis=1) / size_of_window
    time_entropy_averaged = time_entropy[0:np.size(entropy_averaged)]

    return entropy_averaged, time_entropy_averaged

# After that we want to take the differentiation of the average in order to find the whereabouts of maximal changes.
# This function also auto-smoothens. Accepts a parameter to suggest "Smoothness" amount.
# You can specify the order of differentiation by changing o_differ. Default 1.
# Best accepts entropic values that are averaged.
def entropy_differentiator(time_entropy, entropy_averaged, size_of_window = 400, o_differ = 1):

    entropy_arrays_differentiated = np.abs(np.diff(entropy_averaged, o_differ))
    to_average_diff = np.lib.stride_tricks.sliding_window_view(entropy_arrays_differentiated, size_of_window)
    diff_averaged = np.sum(to_average_diff, axis=1) / size_of_window
    time_diff_averaged = time_entropy[0:np.size(diff_averaged)]

    return diff_averaged, time_diff_averaged

# Below is the particular function that takes all the processing time, we compile this using njit.

@njit
def optimised_rve(signal_noised_window_indices, histy_stuff, alpha_filter, all_to_bin_list_sequential, total_bins, max_section_width, samples_per_window):

    entropy_arrays = np.zeros(max_section_width)  # To input values

    for i_x, ordering in enumerate(signal_noised_window_indices[:]):  # Enumerate simply creates for us a counter, i_x

        # alpha_filter practically makes the latest window be the most important.
        histy_stuff *= alpha_filter

        # A discount np.all function
        total_size = samples_per_window
        discount_np_all = np.where(all_to_bin_list_sequential == ordering, 1, 0)
        discount_summed_bins = np.sum(discount_np_all, axis=1)
        discount_isthistrue = np.where(discount_summed_bins == total_size, True, False)

        # Counts all the categories as needed
        histy_stuff[np.where(discount_isthistrue)[0][0]] += 1

        # Normalises and finds the probability of stuff.
        histy_stuff_normalised = histy_stuff / np.sum(histy_stuff)

        # To make sure our NAN does not occur
        histy_stuff_normalised = np.where(histy_stuff_normalised < 10 ** (-75), 10 ** (-75), histy_stuff_normalised)

        # Figuring out shannon entropy.
        sum_p_ln_p = np.sum(-histy_stuff_normalised * np.log(histy_stuff_normalised))
        shannon_entropy = sum_p_ln_p / np.log(total_bins)

        # Plot into i_x the shanon entropy that we get.
        entropy_arrays[i_x] = shannon_entropy

    return entropy_arrays

# Below takes a range of f_crit values -> epsilon and sums up all of the graphs into one big one.

def rve_frequency_averager(time_signal, signal, frequency, f_crits_min = 3, f_crits_max = 20, f_crits_step = 1, τ_const = 0.75, samples_per_window = 5):

    f_crits = np.arange(f_crits_min, f_crits_max, f_crits_step)
    epsilon_step_crits = np.ceil(frequency / (2 * f_crits))
    epsilon_step_crits = epsilon_step_crits  # Removes repeat values. Try one without rmv repeat vals

    # To avoid an int error, we vectorise and then return practically the same values but in int forms.
    epsilon_step_crits = np.vectorize(int)(epsilon_step_crits)

    # Keeps a track of the lowest array size to remove last element entropy bias for the sum, starting from largest
    # possible value.
    lowest_array_size = np.size(signal)
    entropy_arrays_summer = np.zeros(np.size(signal))

    # Now for some multi-processing batch shenanigans that will clearly not be pain. TO BE DONE LATER
    # BATCH_SIZE = epsilon_step_crits // os.cpu_count()  # Hopefully finds the most efficient batch size to compute to this set number.

    for epsilon_step_i in epsilon_step_crits:

        entropy_arrays_temp, time_entropy_temp = rve_of_singal(time_signal, signal, frequency, τ_const, epsilon_step_i, samples_per_window)
        current_array_size = np.size(entropy_arrays_temp)

        entropy_arrays_summer[0:current_array_size] = entropy_arrays_summer[0:current_array_size] + entropy_arrays_temp
        if lowest_array_size > current_array_size:
            lowest_array_size = current_array_size

    # Our final averaged shenanigans, time to plot.
    entropy_arrays_averaged = entropy_arrays_summer[0:lowest_array_size] / np.size(epsilon_step_crits)
    entropy_time_averaged = time_signal[0:lowest_array_size]

    return entropy_arrays_averaged, entropy_time_averaged

def anomalous_point_finder(entropy_signal):
    sorted_entropy_signal = np.sort(entropy_signal, axis=0)
