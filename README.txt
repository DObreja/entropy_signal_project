# Fourier analysis program
This program analyses a set of data.
The frequency and time of the dataset is matched so that a proper window for processing can be detemrined.
A specific critical frequency interval can then be inputted in order to see if a signal lies within that window.
Using an entropy equation allows for further analysis of the signal.
The window is then set manually depending on how precise you want your output dataset to be.

The blue graph is the signal ampltiude over time. Y axis is amplitude, X axis is time.
The red graph is entropy over time. Y Axis is amplitude, X axis is time.

You can find more information on what base settings you can use under `fourier_analysis_program/Fourier_Filtering/instructions.txt`
You can also find under `fourier_analysis_program/Fourier_Filtering/` some example datasets that can be used.

Build the program by making a venv, sourcing into the required environment then running python on main.py .

# Acknowledgements
This project was developed while I was studying at the University of Nottingham.

MIT License
