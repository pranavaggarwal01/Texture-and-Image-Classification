Note: 
- All the programs are implemented with the following details:
For Problem 1 and Problem 2 
Programming language - C++
IDE – SublimeText       //Download link - https://sublimetext.com/2
Compiler – g++   (implemented on Terminal of Ubuntu 16.04)
Operating system – Ubuntu 16.04
OS type – 64 bit
libraries used – OpenCV 3.1.0
	      - extended libraries from opencv_contrib_master for xfeatures2D for SIFT and SURF.
Matlab is used for Problem 1a to plot the sample after PCA.

For Problem 3
Programmed in MATLAB R2016a.
Operating System – Windows 10
OS type – 64 bit
Toolboxes downloaded from https://github.com/pdollar/edges and https://github.com/pdollar/toolbox

Important Notes:
- The folder Texture Classification will contain 3 more folders named Images1 ,Images2 and filtered_images. For Images1 and Image2 each has 4 folders Type1, Type2, Type3 and Type4. This for problem 1a where after the program is executed the textures will automatically will put in their respective folders. Image1 is for 25 dimention feature space and Image2 is for 3 dimenstion after PCA. Care has to be taken to delete the image files in the folders before new execution of the code. The filtered_images folder is used in Problem 1b to save all the 25 filtered images.
- The folder Edge_detection will have a folder called as Edges and this is the folder having the toolboxes got online for the execution of the performance evaluation code. The Edges folder needs to be added in the path where the matlab code will be by first moving the foder to the path and then using the Matlab interface to right click the folder Edge --> Add to path --> All folders and subfolders.


--Before exection of individual files for Texture Classification and Vehicle Classification in C++ in their respective folders, first call the CMakeLists.txt and make the make files and compile the codes by doing the following steps
cmake .
make

once the executables of the problems are made use the below commands to execute them. 

List of files.

-Problem1_a.cpp

This program performs Texture Classification.
The user needs to enter the filename image_width image_height

To execute
./P1a 128 128



-Problem1_b.cpp

This program performs Texture segmentation
The user needs to enter the filename image_name width height
-the user will be asked to enter the mask size and the number of clusters (segments)


To execute
./P1b Comb_1.raw 500 425 

Enter mask_size:
21
Clutter size:
4



-Problem2_a.cpp
This problem gets the keypoints and descriptors
The user needs to enter the filename 


To execute
./P2a 



-Problem2_b_sift.cpp
This problem finds the sift matched between two images
The user needs to enter the filename image_name1 image2

To execute
./P2b_sift jeep.jpg rav4_1.jpg



-Problem2_b_surf.cpp
This problem finds the surf matched between two images
The user needs to enter the filename image_name1 image2

To execute
./P2b_surf jeep.jpg rav4_1.jpg



-Problem2_c_sift.cpp
This problem finds the sift BOW classification
The user needs to enter the filename image_name

To execute
./P2c_sift rav4_2.jpg 



-Problem2_c_surf.cpp
This problem finds the surf BOW classification
The user needs to enter the filename image_name

To execute
./P2c_surf rav4_2.jpg 

For Problem 3
Please implement plot_PCA_points.m, Problem3_Canny.m, Problem3_SE_Zebra.m and Problem3_SE_Jaguar.m in Matlab IDE such as Matlab R2016a is what i used. Only press the run button to execute the programs and also follow the instructions given above regarding adding the Edges folder to the directory and adding its path to the program.
--The get_matrix.m is used to convert .raw file to matrix.
--For finding the F- measure I have used the edgesDemo.m file from github as it was instructed by TA

