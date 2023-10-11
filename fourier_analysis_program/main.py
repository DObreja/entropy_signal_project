import RVE_function

import numpy as np
import matplotlib.pyplot
import sys

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
        self.dynamic_canvas_1 = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.dynamic_canvas_1)
        self.addToolBar(NavigationToolbar(self.dynamic_canvas_1, self))

        # We define the dynamic canvas here
        self.dynamic_canvas_2 = FigureCanvas(Figure(figsize=(5, 3)))
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

        entropy_arrays, entropy_time = RVE_function.rve_of_singal(time_signal, signal, self.frequency, self.τ_const, self.epsilon_step, self.samples_per_window)


        # This part will plot the data by clearing the canvas and replotting
        self.dynamic_axes_1.figure.clf()
        self.dynamic_axes_1 = self.dynamic_canvas_1.figure.subplots()
        self.dynamic_axes_1.plot(time_signal, signal, 'b')
        self.dynamic_axes_1.plot(entropy_time, entropy_arrays, 'r')
        self.dynamic_axes_1.figure.canvas.draw()

        self.dynamic_axes_2.figure.clf()
        self.dynamic_axes_2 = self.dynamic_canvas_2.figure.subplots()
        self.dynamic_axes_2.plot(entropy_time, entropy_arrays, 'r')
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

                self.outer_widget = outer_widget

                # Defines button
                self.dialog_button_save = QtWidgets.QPushButton("Save")
                self.dialog_button_discard = QtWidgets.QPushButton("Cancel")

                # Defines a group box and layout for it, alongside a form layout
                self.dialog_group_box = QtWidgets.QGroupBox("Graph Generation Parameters")
                self.dialog_group_box_offsets = QtWidgets.QGroupBox("Graph Sectioning Parameters")
                self.group_box_layout = QtWidgets.QFormLayout()
                self.group_box_offsets_layout = QtWidgets.QFormLayout()

                # Defines a spinbox
                self.spinbox_duration = QtWidgets.QDoubleSpinBox()
                self.spinbox_frequency = QtWidgets.QSpinBox()
                self.spinbox_τ_const = QtWidgets.QDoubleSpinBox()
                self.spinbox_epsilon_step = QtWidgets.QSpinBox()
                self.spinbox_samples_per_window = QtWidgets.QSpinBox()

                self.spinbox_offset_start = QtWidgets.QSpinBox()
                self.spinbox_offset_finish = QtWidgets.QSpinBox()
                self.checkbox_offset_allow = QtWidgets.QCheckBox("Allow graph offsets?")

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

                # Checks state of checkbox_offset and greys spinbox if false. Also stars off spinbox offset as off
                self.checkbox_offset_allow.stateChanged.connect(self.offsets_disabled)
                self.spinbox_offset_start.setDisabled(True)
                self.spinbox_offset_finish.setDisabled(True)

                # Assigns spinboxes to layout and thus it to group box generation
                self.group_box_layout.addRow("Duration", self.spinbox_duration)
                self.group_box_layout.addRow("Frequency", self.spinbox_frequency)
                self.group_box_layout.addRow("τ constant", self.spinbox_τ_const)
                self.group_box_layout.addRow("Epsilon ε step", self.spinbox_epsilon_step)
                self.group_box_layout.addRow("Samples per window", self.spinbox_samples_per_window)
                self.dialog_group_box.setLayout(self.group_box_layout)

                # Assigns spinboxes to the offsets for the graph.
                self.group_box_offsets_layout.addRow("Start index", self.spinbox_offset_start)
                self.group_box_offsets_layout.addRow("Finish index", self.spinbox_offset_finish)
                self.group_box_offsets_layout.addRow(self.checkbox_offset_allow)
                self.dialog_group_box_offsets.setLayout(self.group_box_offsets_layout)

                # Widget for dialog layout
                dialog_layout = QtWidgets.QVBoxLayout()
                self.setLayout(dialog_layout)

                # Adds the widget for inside the dialog.
                dialog_layout.addWidget(self.dialog_group_box)
                dialog_layout.addWidget(self.dialog_group_box_offsets)

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
                self.accept()

            def cancelling(self):
                self.accept()


        # Exectues the dialog widget
        dialog_init = DialogWidget(self)
        dialog_init.exec_()

# Leftover debugging, probably will be removed later.
def grapher():
    # FREQUENCY
    frequency = 1200
    # FOR THE WINDOW
    epsilon_step = 4  # Values to skip
    samples_per_window = 5  # Total amount of samples that we expect for each window.

    # VALUE DAMPENER, τ_const = magnitude of decay over time.
    τ_const = 4

    # Lets try loading in some noice data
    t_2 = np.linspace(0, 40, int(40 * 1200))
    signal_noised_y = np.loadtxt("Fourier_Filtering\signal.mat")
    amplitude = np.max(signal_noised_y)

    entropy_arrays, time_entropy = RVE_function.rve_of_singal(t_2, signal_noised_y, frequency, τ_const, epsilon_step, samples_per_window)
    # Now lets plot a graph which has the signal and the entropy superimposed.
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(t_2, signal_noised_y)
    matplotlib.pyplot.plot(time_entropy, entropy_arrays * amplitude - amplitude)
    matplotlib.pyplot.show()
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(time_entropy, entropy_arrays)
    matplotlib.pyplot.show()

if __name__ == "__main__":

    # grapher()

    app = QtWidgets.QApplication([])

    widget = TheWidget()
    widget.resize(1200, 800)
    widget.show()

    sys.exit(app.exec_())