// Problem 2a - Extraction and description of salient points. Created by Pranav Aggarwal

#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char *argv[])

{	

	Mat jeep = cv::imread("jeep.jpg",CV_LOAD_IMAGE_COLOR);
	Mat bus = cv::imread("bus.jpg",CV_LOAD_IMAGE_COLOR);

  cv::imshow( "jeep", jeep ); //displaying the images
   
  cv::imshow("bus", bus);

  // performing SIFT
  cv::Mat jeep_edges(jeep.size(),CV_8U);

  cv::Ptr<Feature2D> sift_feature = xfeatures2d::SIFT::create();
  std::vector<KeyPoint> keypoints_1, keypoints_2;    // creating template of datatype keypoints
  sift_feature->detect( jeep, keypoints_1 );  // finding the keypoints
  Mat descriptors_1, descriptors_2;    	
	sift_feature->compute( jeep, keypoints_1, descriptors_1 ); //extracting the features from the keypoints
	sift_feature->detectAndCompute(bus, cv::noArray(), keypoints_2,descriptors_2, false);
	cout<< descriptors_1.size()<<endl; //printing the dimention of descriptors
	cout<< descriptors_2.size()<<endl;
/*	BFMatcher matcher(NORM_L2  , true);
  vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_1, matches );
*/

  drawKeypoints(jeep, keypoints_1, jeep_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("keypoints_sift1",jeep_edges);   //displaying the keypoints on the images
	imwrite("keypoint_sift_jeep.png",jeep_edges);	
  cv::Mat bus_edges(bus.size(),CV_8U);
  drawKeypoints(bus, keypoints_2, bus_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("keypoints_sift2",bus_edges);
  imwrite("keypoint_sift_bus.png",bus_edges);

  // performaing SURF 

  cv::Ptr<Feature2D> surf_feature = xfeatures2d::SURF::create();
  surf_feature->detect( jeep, keypoints_1 );  
	surf_feature->compute( jeep, keypoints_1, descriptors_1 );
	surf_feature->detectAndCompute(bus, cv::noArray(), keypoints_2,descriptors_2, false);
	cout<< descriptors_1.size()<<endl; //printing the dimention of descriptors
	cout<< descriptors_2.size()<<endl;
	
  drawKeypoints(jeep, keypoints_1, jeep_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("keypoints_surf1",jeep_edges);
	imwrite("keypoint_surf_jeep.png",jeep_edges);	

  drawKeypoints(bus, keypoints_2, bus_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("keypoints_surf2",bus_edges);
  imwrite("keypoint_surf_bus.png",bus_edges);

  cv::waitKey(0); 
  return 0;
}