# ADALINE-Network

## Introduction 
Implementation of Widrow and Hoff's ADALINE network for pattern recognition on an 8x8 matrix representing the letters A, D, O, Q, G, V and W. 

The use of the network is explored and the recognition is improved by adding a shift to the left and right letters.

Finally, some graphs are shown to observe the behavior of the network after some epochs with different values of the learning rate and the minimum error.

## Technologies
* Matlab

## Usage
Using Matlab, import and run the program. First, type "data.txt" and "target.txt" that corresponds to the followinf file names.

- The file data.txt contains the data that represents the letters into the matrix
- The file target.txt contains the target letters showing as numbers

Then, type the values epochMAx, Eepoch and alfa will be asked.

- epochMax is the max number of epochs before stop, example 100.
- Eepoch is the minimum error expected before stop, example 0.3.
- alfa is the learning rate, example 0.01.

## ScreenShots 
![A](/ScreenShots/datosexcel0.PNG)

![B](/ScreenShots/ml5.PNG)
