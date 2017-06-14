// Problem 2b Image Matching using surf Created by Pranav Aggarwal
// The user has to enter the name of both the images that have to be mapped.
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
double setTreshold(int , int );

int main(int argc, char *argv[])

{
  //Reading file

  Mat vehical1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR); //loading the images
  Mat vehical2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);

  cv::imshow( "image1", vehical1 );
   
  cv::imshow("image2", vehical2);

  cv::Mat vehical1_edges(vehical1.size(),CV_8U);
  cv::Mat vehical2_edges(vehical2.size(),CV_8U);
  cv::Mat rav_matches1;
  cv::Mat rav_matches2;

 
  cv::Ptr<Feature2D> feature_surf = xfeatures2d::SURF::create();
  std::vector<KeyPoint> keypoints_1, keypoints_2, selected_keypoints1, selected_keypoints2;    

  Mat descriptors_1, descriptors_2;
  // Finfing the keypoints and descriptors    
  feature_surf->detectAndCompute(vehical1, cv::noArray(), keypoints_1,descriptors_1, false);
  feature_surf->detectAndCompute(vehical2, cv::noArray(), keypoints_2,descriptors_2, false);
  double min, max;
  int min_loc;
  int count = 0;
  //will compare the best 25 features
  int best_des1[25];
  int best_des2[25];
  Mat dist_bet_des(descriptors_2.rows,1, CV_32FC1);
  double thres = setTreshold(vehical1.rows,vehical2.rows);
  for (int i = 0; i < descriptors_1.rows; i++)
  {
    for (int j = 0; j < descriptors_2.rows; j++)
    {
      dist_bet_des.at<float>(j) = norm(descriptors_1.row(i), descriptors_2.row(j), NORM_L2, cv::noArray() );
    }

  cv::minMaxLoc(dist_bet_des, &min, &max);
  for(int s = 0; s < descriptors_2.rows; s++)
  {
    if (dist_bet_des.at<float>(s) == min)
    {
      min_loc = s;
      //printf("%d     %d\n", s,i);
      break;
    }
  }
  //printf("%f\n", min);
  if(min < thres)
  {
    best_des1[count] = i;
    best_des2[count] = min_loc;
    //cout << min_loc;
    //printf("%d\n",min_loc);
    count++;
  }
  if (count == 25)
  {break;}
  }

  Mat selected_des1(25, 128,CV_32FC1);
  Mat selected_des2(25, 128,CV_32FC1);
  for (int i = 0; i < 25; i++)
  {
    for (int j = 0; j < 128; j++)
    {
      selected_des1.at<float>(i,j) = descriptors_1.at<float>(best_des1[i],j);
      selected_des2.at<float>(i,j) = descriptors_2.at<float>(best_des2[i],j);
    }
    /*selected_keypoints1[i].pt = keypoints_1[best_des1[i]].pt;
    selected_keypoints2[i].pt = keypoints_2[best_des2[i]].pt;*/
    
  }


  BFMatcher bf_obj(NORM_L2  , true); //using BruteForce to match the images
  vector< DMatch > matches;
  bf_obj.match( descriptors_1, descriptors_2, matches );
 /* vector< vector<DMatch> > matches;
  Ptr<DescriptorMatcher> bf_obj = DescriptorMatcher::create("BruteForce");
  bf_obj->knnMatch( descriptors_1, descriptors_2, matches, 500 );*/

  //cv::DescriptorMatcher matcher;

  //std::cout << descriptors_1;

  //will mark the keypoints
  drawKeypoints(vehical1, keypoints_1, vehical1_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints(vehical2, keypoints_2, vehical2_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("keypoints_1",vehical1_edges);
  imshow("keypoints_2",vehical2_edges);

  //vector<Dmatch>::const_iterator first = matches.begin();
  //vector<Dmatch>::const_iterator last = matches.begin() + 10;
  //vector<DMatch> matches_count(matches.begin(), matches.begin() + 10);  
  // will draw the matches
  drawMatches(vehical1, keypoints_1, vehical2, keypoints_2, matches, rav_matches2, Scalar::all(-1));  
  imshow("matches",rav_matches2);
  imwrite("all_matches_surf.png",rav_matches2);


  //drawing only the important 25 matches
  // Formating the visualization of the images
  if (vehical1.rows > vehical2.rows)
  {
    cv::Mat mat_temp1(vehical1.rows - vehical2.rows,vehical2.cols,CV_8UC3);
    mat_temp1 = Scalar(5);
    cout << mat_temp1.size();
    //cout << 
    cv::Mat mat_temp2;
    cv::vconcat( vehical2, mat_temp1, mat_temp2);
    cv::hconcat(vehical1, mat_temp2, rav_matches1);
  }
  else if (vehical1.rows < vehical2.rows)
  {
    cv::Mat mat_temp1(vehical2.rows - vehical1.rows,vehical1.cols,CV_8UC3);
    mat_temp1 = Scalar(5);
    //cout << mat_temp1.size();
    cv::Mat mat_temp2;
    cv::vconcat(vehical1, mat_temp1, mat_temp2);
    cv::hconcat(mat_temp2, vehical2, rav_matches1);
  }
  else
  {
    cv::hconcat(vehical1, vehical2, rav_matches1);
  }
  
  vector<Point2f> image1pts;
    vector<Point2f> image2pts;
    
    for (int i = 0; i < 25; i++)
    {
      KeyPoint pt1 = keypoints_1[best_des1[i]];
      KeyPoint pt2 = keypoints_2[best_des2[i]];
      image1pts.push_back(pt1.pt);
      image2pts.push_back(pt2.pt);
    }
    for (int i = 0; i < 25; i++) {
      Point2f pt1 = image1pts[i];
      Point2f pt2 = image2pts[i];
      Point2f from = pt1;
      Point2f to   = Point(500 + pt2.x, pt2.y);
      line(rav_matches1, from, to, Scalar(10*(25-i), 10*i, 255*(i % 2)));
      circle(rav_matches1, pt1, cvRound(5), Scalar(0,0,0), 1, 8, 0);
      circle(rav_matches1, Point(500 + pt2.x, pt2.y), cvRound(5), Scalar(0,0,0), 1, 8, 0);
    }
    cv::imshow("Important_features", rav_matches1);
    imwrite("best_matches_surf.png",rav_matches1);

    cv::waitKey(0); 
    return 0;
}
double setTreshold(int r1, int r2)
{
  if(r1 == 380 || r2 == 380)
  {return 0.200;} // for jeep
  else if (r1 == 240 || r2 == 240)
  {return 0.250;} //for bus
  else
  {return 0.50;} //for rav

}


