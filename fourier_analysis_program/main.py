import RVE_function

import numpy as np
import matplotlib.pyplot
import sys

import time

from PySide2 import QtCore, QtWidgets, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

class TheWidget(QtWidgets.QMainWindow):
    def __init__(self):

        # Initialised variables that will be critical for changing the graph shape.
        self.file_path = ""
        self.duration = 40
        self.frequency = 1200
        self.τ_const = 1
        self.epsilon_step = 4
        self.samples_per_window = 5

        self.offset_start = 0
        self.offset_finish = int(self.duration*self.frequency)
        self.allow_offset = False

        self.allow_composite = False
        self.frequency_crits_min = 5
        self.frequency_crits_max = 40
        self.frequency_crits_step = 1

        self.allow_smoothening = False
        self.show_differentiated_graph = False

        # Inherits methods and properties from another class.
        super().__init__()

        # Defines a push button.
        self.button_start = QtWidgets.QPushButton("Start Graph")
        self.button_open = QtWidgets.QPushButton("Open")
        self.button_parameters = QtWidgets.QPushButton("Edit Parameters")

        # Defines a text box for file selected
        self.text_box = QtWidgets.QLabel("File Path: None")

        # Widget for box layout
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # We define the static canvas here
        self.dynamic_canvas_1 = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(self.dynamic_canvas_1)
        self.addToolBar(NavigationToolbar(self.dynamic_canvas_1, self))

        # We define the dynamic canvas here
        self.dynamic_canvas_2 = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(self.dynamic_canvas_2)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.dynamic_canvas_2, self))

        # We initialise the axes here.
        self.dynamic_axes_1 = self.dynamic_canvas_1.figure.subplots()
        self.dynamic_axes_2 = self.dynamic_canvas_2.figure.subplots()

        # Create a widget in layout to create a sub layout. This structures the layouts to be vertical in order.
        widget_sub = QtWidgets.QWidget()
        layout.addWidget(widget_sub)
        layout_sub = QtWidgets.QHBoxLayout(widget_sub)

        # Adds a push button
        layout_sub.addWidget(self.button_start)
        layout_sub.addWidget(self.button_open)
        layout_sub.addWidget(self.button_parameters)

        # Adds a text button
        layout.addWidget(self.text_box)

        # When start button clicked, run a function
        self.button_start.clicked.connect(self.begin_graph)
        self.button_open.clicked.connect(self.select_file)
        self.button_parameters.clicked.connect(self.parameter_edit)

        # Sets the central widget
        self.setCentralWidget(widget)

    # This function runs main.
    def begin_graph(self):
        # Prepares all the data for input.
        time_duration = self.duration
        time_points = int(self.duration * self.frequency)
        time_signal = np.linspace(0, time_duration, time_points)

        signal = np.loadtxt(self.file_path[0])

        # This will section the graph (If needed at all)
        if self.allow_offset == True:
            time_signal = time_signal[self.offset_start:self.offset_finish]
            signal = signal[self.offset_start:self.offset_finish]

        # This will take a composite of the graph.
        if self.allow_composite == True:
            entropy_arrays, entropy_time = RVE_function.rve_frequency_averager(time_signal, signal, self.frequency, self.frequency_crits_min, self.frequency_crits_max, self.frequency_crits_step, self.τ_const, self.samples_per_window)
        else:
            entropy_arrays, entropy_time = RVE_function.rve_of_singal(time_signal, signal, self.frequency, self.τ_const,
                                                                      self.epsilon_step, self.samples_per_window)

        # If certain conditions are met, this will further smoothen the graph.
        if self.allow_smoothening == True:
            entropy_arrays, entropy_time = RVE_function.entropy_window_averager(entropy_time, entropy_arrays)

        # If certain conditions are met, generates an external figure that shows a differentiated graph.
        if self.show_differentiated_graph == True:
            diff_entropy_arrays, diff_entropy_time = RVE_function.entropy_differentiator(entropy_time, entropy_arrays)
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(time_signal, signal, 'b')
            matplotlib.pyplot.title("Signal over an amount of time.")
            matplotlib.pyplot.ylabel("Amplitude of signal")
            matplotlib.pyplot.xlabel("Time ($s$)")
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(entropy_time, entropy_arrays, 'r')
            matplotlib.pyplot.title("The entropy of a signal over time.")
            matplotlib.pyplot.ylabel("Entropy of signal")
            matplotlib.pyplot.xlabel("Time ($s$)")
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(diff_entropy_time, diff_entropy_arrays, 'g')
            matplotlib.pyplot.title("First order differential of the entropy over time.")
            matplotlib.pyplot.ylabel("Entropy change of signal")
            matplotlib.pyplot.xlabel("Time ($s$)")

            # DEBUG SHTUFF: fft graph for showing how to get vals.
            signal_fft = np.fft.fft(signal)
            time_fft = np.fft.fftfreq(np.size(signal_fft), 1/self.frequency)
            time_fft_shifted = np.fft.fftshift(time_fft)
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(time_fft, signal_fft)
            matplotlib.pyplot.title("Fast fourier transform of the brain signal")
            matplotlib.pyplot.ylabel("Amplitude of the fast fourier")
            matplotlib.pyplot.xlabel("Frequency ($hz$)")
            matplotlib.pyplot.show()

        # This part will plot the data by clearing the canvas and replotting
        self.dynamic_axes_1.figure.clf()
        self.dynamic_axes_1 = self.dynamic_canvas_1.figure.subplots()
        self.dynamic_axes_1.plot(time_signal, signal, 'b')
        self.dynamic_axes_1.plot(entropy_time, entropy_arrays*np.max(signal)*3 - np.max(signal)*2.5, 'r')
        self.dynamic_axes_1.set_title('A graph of the signal and the entropy over time, with y axes adjusted to fit')
        self.dynamic_axes_1.set_xlabel('Time (s)')
        self.dynamic_axes_1.set_ylabel('Amplitude')
        self.dynamic_axes_1.figure.canvas.draw()

        self.dynamic_axes_2.figure.clf()
        self.dynamic_axes_2 = self.dynamic_canvas_2.figure.subplots()
        self.dynamic_axes_2.plot(entropy_time, entropy_arrays, 'r')
        self.dynamic_axes_2.set_title('A graph of the entropy over time.')
        self.dynamic_axes_2.set_xlabel('Time (s)')
        self.dynamic_axes_2.set_ylabel('Amplitude')
        self.dynamic_axes_2.figure.canvas.draw()



    # This function will run a file explorer to select a file
    def select_file(self):
        self.file_path = QtWidgets.QFileDialog.getOpenFileName(self, "Open mat file", "", "Mat Files (*.mat)")
        self.text_box.setText(f"File Path: {self.file_path[0]}")

    # This function will open a dialog box to edit some parameters.
    def parameter_edit(self):

        # Supposedly have to create a class for the box widget
        class DialogWidget(QtWidgets.QDialog):
            def __init__(self, outer_widget):
                # Inherits methods and properties from another class.
                super().__init__()

                # Displays initial unsaved values for the graph
                self.dummy_duration = outer_widget.duration
                self.dummy_frequency = outer_widget.frequency
                self.dummy_τ_const = outer_widget.τ_const
                self.dummy_epsilon_step = outer_widget.epsilon_step
                self.dummy_samples_per_window = outer_widget.samples_per_window

                self.dummy_offset_start = outer_widget.offset_start
                self.dummy_offset_finish = outer_widget.offset_finish
                self.dummy_allow_offset = outer_widget.allow_offset

                self.dummy_allow_composite = outer_widget.allow_composite
                self.dummy_frequency_crits_min = outer_widget.frequency_crits_min
                self.dummy_frequency_crits_max = outer_widget.frequency_crits_max
                self.dummy_frequency_crits_step = outer_widget.frequency_crits_step

                self.dummy_allow_smoothening = outer_widget.allow_smoothening
                self.dummy_show_differentiated_graph = outer_widget.show_differentiated_graph

                self.outer_widget = outer_widget

                # Defines button
                self.dialog_button_save = QtWidgets.QPushButton("Save")
                self.dialog_button_discard = QtWidgets.QPushButton("Cancel")

                # Defines a group box and layout for it, alongside a form layout
                self.dialog_group_box = QtWidgets.QGroupBox("Graph Generation Parameters")
                self.dialog_group_box_offsets = QtWidgets.QGroupBox("Graph Sectioning Parameters")
                self.dialog_group_box_composite = QtWidgets.QGroupBox("Graph Composite Stacking Parameters")
                self.dialog_group_box_graph_misc = QtWidgets.QGroupBox("Graph Additional Analysis Features")
                self.group_box_layout = QtWidgets.QFormLayout()
                self.group_box_offsets_layout = QtWidgets.QFormLayout()
                self.group_box_composite_layout = QtWidgets.QFormLayout()
                self.group_box_graph_misc_layout = QtWidgets.QFormLayout()

                # Defines a spinbox and checkboxes
                self.spinbox_duration = QtWidgets.QDoubleSpinBox()
                self.spinbox_frequency = QtWidgets.QSpinBox()
                self.spinbox_τ_const = QtWidgets.QDoubleSpinBox()
                self.spinbox_epsilon_step = QtWidgets.QSpinBox()
                self.spinbox_samples_per_window = QtWidgets.QSpinBox()

                self.spinbox_offset_start = QtWidgets.QSpinBox()
                self.spinbox_offset_finish = QtWidgets.QSpinBox()
                self.checkbox_offset_allow = QtWidgets.QCheckBox("Allow graph offsets?")

                self.checkbox_composite_allow = QtWidgets.QCheckBox("Allow composite graph?")
                self.spinbox_frequency_crits_min = QtWidgets.QSpinBox()
                self.spinbox_frequency_crits_max = QtWidgets.QSpinBox()
                self.spinbox_frequency_crits_step = QtWidgets.QSpinBox()

                self.checkbox_allow_smoothening = QtWidgets.QCheckBox("Allow entropy smoothening?")
                self.checkbox_show_differentiated_graph = QtWidgets.QCheckBox("Show differentiated graph?")

                # Defines spinbox and checkbox default values
                self.spinbox_duration.setMaximum(1000000)
                self.spinbox_duration.setMinimum(0)
                self.spinbox_duration.setSingleStep(0.01)
                self.spinbox_duration.setValue(self.dummy_duration)

                self.spinbox_frequency.setMaximum(1000000)
                self.spinbox_frequency.setMinimum(0)
                self.spinbox_frequency.setSingleStep(1)
                self.spinbox_frequency.setValue(self.dummy_frequency)

                self.spinbox_τ_const.setMaximum(1000000)
                self.spinbox_τ_const.setMinimum(0)
                self.spinbox_τ_const.setSingleStep(0.01)
                self.spinbox_τ_const.setValue(self.dummy_τ_const)

                self.spinbox_epsilon_step.setMaximum(1000000)
                self.spinbox_epsilon_step.setMinimum(0)
                self.spinbox_epsilon_step.setSingleStep(1)
                self.spinbox_epsilon_step.setValue(self.dummy_epsilon_step)

                self.spinbox_samples_per_window.setMaximum(1000000)
                self.spinbox_samples_per_window.setMinimum(0)
                self.spinbox_samples_per_window.setSingleStep(1)
                self.spinbox_samples_per_window.setValue(self.dummy_samples_per_window)

                self.spinbox_offset_start.setMaximum(int(self.dummy_frequency*self.dummy_duration))
                self.spinbox_offset_start.setMinimum(0)
                self.spinbox_offset_start.setSingleStep(1)
                self.spinbox_offset_start.setValue(self.dummy_offset_start)

                self.spinbox_offset_finish.setMaximum(int(self.dummy_frequency*self.dummy_duration))
                self.spinbox_offset_finish.setMinimum(0)
                self.spinbox_offset_finish.setSingleStep(1)
                self.spinbox_offset_finish.setValue(self.dummy_offset_finish)

                self.checkbox_offset_allow.setChecked(self.dummy_allow_offset)

                self.checkbox_composite_allow.setChecked(self.dummy_allow_composite)

                self.spinbox_frequency_crits_min.setMaximum(300)
                self.spinbox_frequency_crits_min.setMinimum(1)
                self.spinbox_frequency_crits_min.setSingleStep(1)
                self.spinbox_frequency_crits_min.setValue(self.dummy_frequency_crits_min)

                self.spinbox_frequency_crits_max.setMaximum(300)
                self.spinbox_frequency_crits_max.setMinimum(1)
                self.spinbox_frequency_crits_max.setSingleStep(1)
                self.spinbox_frequency_crits_max.setValue(self.dummy_frequency_crits_max)

                self.spinbox_frequency_crits_step.setMaximum(10)
                self.spinbox_frequency_crits_step.setMinimum(1)
                self.spinbox_frequency_crits_step.setSingleStep(1)
                self.spinbox_frequency_crits_step.setValue(self.dummy_frequency_crits_step)

                self.checkbox_allow_smoothening.setChecked(self.dummy_allow_smoothening)

                self.checkbox_show_differentiated_graph.setChecked(self.dummy_show_differentiated_graph)

                # Checks state of checkbox_offset and greys spinbox if false. Also starts off spinbox offset as off
                self.checkbox_offset_allow.stateChanged.connect(self.offsets_disabled)
                self.spinbox_offset_start.setDisabled(not self.dummy_allow_offset)
                self.spinbox_offset_finish.setDisabled(not self.dummy_allow_offset)

                # Checks state of composite_allow and greys step if True.
                self.checkbox_composite_allow.stateChanged.connect(self.composite_disabled)
                self.spinbox_frequency_crits_min.setDisabled(not self.dummy_allow_composite)
                self.spinbox_frequency_crits_max.setDisabled(not self.dummy_allow_composite)
                self.spinbox_frequency_crits_step.setDisabled(not self.dummy_allow_composite)

                # Assigns spinboxes to layout and thus it to group box generation
                self.group_box_layout.addRow("Duration", self.spinbox_duration)
                self.group_box_layout.addRow("Frequency", self.spinbox_frequency)
                self.group_box_layout.addRow("τ constant", self.spinbox_τ_const)
                self.group_box_layout.addRow("Epsilon ε step", self.spinbox_epsilon_step)
                self.group_box_layout.addRow("Samples per window", self.spinbox_samples_per_window)
                self.dialog_group_box.setLayout(self.group_box_layout)

                # Assigns spinboxes and checkboxes to the offsets for the graph.
                self.group_box_offsets_layout.addRow("Start index", self.spinbox_offset_start)
                self.group_box_offsets_layout.addRow("Finish index", self.spinbox_offset_finish)
                self.group_box_offsets_layout.addRow(self.checkbox_offset_allow)
                self.dialog_group_box_offsets.setLayout(self.group_box_offsets_layout)

                # Assigns spinboxes and checkboxes to the timelapse.
                self.group_box_composite_layout.addRow("Crit Frequency Start", self.spinbox_frequency_crits_min)
                self.group_box_composite_layout.addRow("Crit Frequency End", self.spinbox_frequency_crits_max)
                self.group_box_composite_layout.addRow("Crit Frequency Step", self.spinbox_frequency_crits_step)
                self.group_box_composite_layout.addRow(self.checkbox_composite_allow)
                self.dialog_group_box_composite.setLayout(self.group_box_composite_layout)

                # Assigns checkboxes to the miscellaneous features.
                self.group_box_graph_misc_layout.addRow(self.checkbox_allow_smoothening)
                self.group_box_graph_misc_layout.addRow(self.checkbox_show_differentiated_graph)
                self.dialog_group_box_graph_misc.setLayout(self.group_box_graph_misc_layout)

                # Widget for dialog layout
                dialog_layout = QtWidgets.QVBoxLayout()
                self.setLayout(dialog_layout)

                # Adds the widget for inside the dialog.
                dialog_layout.addWidget(self.dialog_group_box)
                dialog_layout.addWidget(self.dialog_group_box_offsets)
                dialog_layout.addWidget(self.dialog_group_box_composite)
                dialog_layout.addWidget(self.dialog_group_box_graph_misc)

                # Adding a dialog sub widget
                dialog_sub_widget = QtWidgets.QWidget()
                dialog_layout.addWidget(dialog_sub_widget)
                dialog_sub_layout = QtWidgets.QHBoxLayout(dialog_sub_widget)

                # Adds the sub_widget
                dialog_sub_layout.addWidget(self.dialog_button_discard)
                dialog_sub_layout.addWidget(self.dialog_button_save)

                # Connects save and cancel
                self.dialog_button_save.clicked.connect(self.saving)
                self.dialog_button_discard.clicked.connect(self.cancelling)


            def offsets_disabled(self):
                self.spinbox_offset_start.setDisabled(not self.checkbox_offset_allow.isChecked())
                self.spinbox_offset_finish.setDisabled(not self.checkbox_offset_allow.isChecked())

            def composite_disabled(self):
                self.spinbox_epsilon_step.setDisabled(self.checkbox_composite_allow.isChecked())
                self.spinbox_frequency_crits_min.setDisabled(not self.checkbox_composite_allow.isChecked())
                self.spinbox_frequency_crits_max.setDisabled(not self.checkbox_composite_allow.isChecked())
                self.spinbox_frequency_crits_step.setDisabled(not self.checkbox_composite_allow.isChecked())


            def saving(self):
                self.outer_widget.duration = self.spinbox_duration.value()
                self.outer_widget.frequency = self.spinbox_frequency.value()
                self.outer_widget.τ_const = self.spinbox_τ_const.value()
                self.outer_widget.epsilon_step = self.spinbox_epsilon_step.value()
                self.outer_widget.samples_per_window = self.spinbox_samples_per_window.value()

                # See if we need to set our graph to the default offset values or no.
                self.outer_widget.allow_offset = self.checkbox_offset_allow.isChecked()
                if self.checkbox_offset_allow.isChecked() == True:
                    self.outer_widget.offset_start = self.spinbox_offset_start.value()
                    self.outer_widget.offset_finish = self.spinbox_offset_finish.value()
                else:
                    self.outer_widget.offset_start = 0
                    self.outer_widget.offset_finish = int(self.dummy_frequency*self.dummy_duration)

                # For our composite area
                self.outer_widget.allow_composite = self.checkbox_composite_allow.isChecked()
                self.outer_widget.frequency_crits_min = self.spinbox_frequency_crits_min.value()
                self.outer_widget.frequency_crits_max = self.spinbox_frequency_crits_max.value()
                self.outer_widget.frequency_crits_step = self.spinbox_frequency_crits_step.value()

                # For our misc area
                self.outer_widget.allow_smoothening = self.checkbox_allow_smoothening.isChecked()
                self.outer_widget.show_differentiated_graph = self.checkbox_show_differentiated_graph.isChecked()

                self.accept()

            def cancelling(self):
                self.accept()


        # Exectues the dialog widget
        dialog_init = DialogWidget(self)
        dialog_init.exec_()

# Leftover debugging, probably will be removed later.
def grapher():
    # FREQUENCY
    frequency = 600
    # FOR THE WINDOW
    epsilon_step = 60  # Values to skip. The critical values to skip is unique to this equation:

    samples_per_window = 5  # Total amount of samples that we expect for each window.

    # VALUE DAMPENER, τ_const = magnitude of decay over time.
    τ_const = 0.75

    # TOTAL TIME
    tot_time = 600

    # Lets try loading in some noice data
    t_2 = np.linspace(0, tot_time, int(tot_time * frequency))
    signal_noised_y = np.loadtxt("Brain_Scanning\data_1_actual.mat")
    signal_noised_y = np.sum(signal_noised_y, axis=0)
    amplitude = np.max(signal_noised_y)

    fft_signal = np.fft.fft(signal_noised_y)
    fft_time = np.fft.fftfreq(360000, 1 / 600)
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(fft_time, fft_signal)
    matplotlib.pyplot.title("Fast fourier transform of the signal over frequency.")
    matplotlib.pyplot.ylabel("Ampltiude")
    matplotlib.pyplot.xlabel("Frequency ($Hz$)")
    matplotlib.pyplot.show()

    # entire_signal_entropied = np.zeros([np.size(signal_noised_y, axis=0), np.size(signal_noised_y, axis=1)])
    # max_y_count = np.size(signal_noised_y, axis=0)
    # max_x_count = np.size(signal_noised_y, axis=1)
    # y_count = 0
    # while y_count < max_y_count:
    #     entropy_arrays, time_entropy = RVE_function.rve_of_singal(t_2, signal_noised_y[y_count, :], frequency, τ_const,
    #                                                               epsilon_step, samples_per_window)
    #     entire_signal_entropied[y_count, :] = entropy_arrays

    entropy_arrays, time_entropy = RVE_function.rve_of_singal(t_2, signal_noised_y, frequency, τ_const, epsilon_step, samples_per_window)
    entropy_averaged, time_entropy_averaged = RVE_function.entropy_window_averager(t_2, entropy_arrays)
    diff_averaged, time_diff_averaged = RVE_function.entropy_differentiator(time_entropy_averaged, entropy_averaged)

    # EXPERIMENTATION
    # Lets do an experiment with a large amount of crit freqs this time. Lets say 5->40 crit freqs
    clocker_start_2 = time.time()
    entropy_arrays_averaged, entropy_time_averaged = RVE_function.rve_frequency_averager(t_2, signal_noised_y, frequency)
    clocker_end_2 = time.time()
    print(clocker_end_2 - clocker_start_2)

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(entropy_time_averaged, entropy_arrays_averaged, 'r')
    matplotlib.pyplot.title("The averaged entropy of a signal over time.")
    matplotlib.pyplot.ylabel("Entropy of signal")
    matplotlib.pyplot.xlabel("Time ($s$)")
    matplotlib.pyplot.show()

    # Now lets plot a graph which has the signal and the entropy superimposed.
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(t_2, signal_noised_y, 'b')
    matplotlib.pyplot.title("The signal over time.")
    matplotlib.pyplot.ylabel("Amplitude")
    matplotlib.pyplot.xlabel("Time ($s$)")
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(time_entropy_averaged, entropy_averaged, 'r')
    matplotlib.pyplot.title("The entropy over time.")
    matplotlib.pyplot.ylabel("Amplitude of Entropy")
    matplotlib.pyplot.xlabel("Time ($s$)")
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(time_diff_averaged, diff_averaged, 'g')
    matplotlib.pyplot.title("The differentiated entropy  over time.")
    matplotlib.pyplot.ylabel("Amplitude of differential")
    matplotlib.pyplot.xlabel("Time ($s$)")
    matplotlib.pyplot.show()

if __name__ == "__main__":

    app = QtWidgets.QApplication([])

    widget = TheWidget()
    widget.resize(1200, 800)
    widget.show()

    sys.exit(app.exec_())
