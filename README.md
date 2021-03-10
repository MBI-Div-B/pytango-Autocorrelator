# pytango-Autocorrelator
This device returns in femtoseconds the pulse duration of the laser.
To select which is the camera connected to the autocorrelator, one must write in dev_properties from the Basler device in Jive, the serial number of the desired camera, which is usually written on the camera as S/N). Then this code will automatically receive a matrix with all the values (the image itself) and one will be able to calculate the FWHM in x and y under different situations.
This code runs in python 3
