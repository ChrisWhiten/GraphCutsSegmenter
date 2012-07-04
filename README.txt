Binary Image Segmentation with Graph Cuts
Chris Whiten

RUNNING THIS IMPLEMENTATION
------------------------------
This project has been built and tested with Visual Studio 2010.
It has a dependency on OpenCV, specifically tested with OpenCV 2.3.1
To run this, open GraphCuts.sln with VS2010 and run from within Visual Studio.

TUNABLE PARAMETERS
------------------------------
To load a specific image, modify the variable 'IMAGE_PATH' near the top of main.cpp.
To change the radius of pixels selected by a mouse click, modify MOUSE_RADIUS (main.cpp).
To change the number of bins in the constructed histograms, modify HISTOGRAM_BINS (main.cpp).
To change the estimated camera noise parameter, modify SIGMA (main.cpp).
To change the relative weighting between the boundary and region term, modify LAMBDA (main.cpp).

USE CASE
------------------------------
Modify IMAGE_PATH and load your desired image to be segmented.
Click and drag, with the left mouse button, over some background pixels.
Click and drag, with the right mouse button, over some foreground pixels.
Multiple strokes can be provided.
Hit 'enter' to view the segmentation