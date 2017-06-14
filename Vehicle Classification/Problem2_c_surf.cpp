// Problem 1c using surf implement BOW classification Created by Pranav Aggarwal 
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
double max(double [8]);

int main(int argc, char *argv[])

{
	

  Mat rav4_2 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
  Mat jeep = cv::imread("jeep.jpg",CV_LOAD_IMAGE_COLOR);
  Mat bus = cv::imread("bus.jpg",CV_LOAD_IMAGE_COLOR);
  Mat rav4_1 = cv::imread("rav4_1.jpg",CV_LOAD_IMAGE_COLOR);

  cv::imshow( "rav4_2", rav4_2 );
   
  //cv::imshow("rav4_2", rav4_2);

/*  cv::Mat rav4_1_edges(rav4_1.size(),CV_8U);
  cv::Mat rav4_2_edges(rav4_2.size(),CV_8U);
  cv::Mat rav_matches(rav4_2.size(),CV_8UC3);*/

  cv::Ptr<Feature2D> surf_features = xfeatures2d::SURF::create();
  std::vector<KeyPoint> keypoints_1, keypoints_2,keypoints_3,keypoints_4;    

  Mat descriptors_1, descriptors_2,descriptors_3,descriptors_4, descriptors_group1,descriptors_group;    
  surf_features->detectAndCompute(rav4_2, cv::noArray(), keypoints_1,descriptors_1, false);
  surf_features->detectAndCompute(rav4_1, cv::noArray(), keypoints_2,descriptors_2, false);
  surf_features->detectAndCompute(jeep, cv::noArray(), keypoints_3,descriptors_3, false);
  surf_features->detectAndCompute(bus, cv::noArray(), keypoints_4,descriptors_4, false);

  cv::vconcat(descriptors_2, descriptors_3, descriptors_group1);
  cv::vconcat(descriptors_group1, descriptors_4, descriptors_group);
  //transpose(descriptors_group,descriptors_group);
  cv::Mat labels,center;
  //cv::Mat centers(8, 1, CV_32FC1);
  cv::kmeans(descriptors_group, 8, labels,
    cv::TermCriteria(CV_TERMCRIT_ITER, 10, 1.0),
      100, cv::KMEANS_RANDOM_CENTERS,center);
  cv::Size s1 = descriptors_1.size();
  cv::Size s2 = descriptors_2.size();
  cv::Size s3 = descriptors_3.size();
  cv::Size s4 = descriptors_4.size();
  
  //double dist1[s1.height];
  double dist[8];
  double max_value;
  int rav4_2_array[8] = {0};
  for (int i = 0; i < s1.height; i++)
  {    
    for (int j = 0; j < 8; j++)
    {
      dist[j] = norm(descriptors_1.row(i), center.row(j), NORM_L2, cv::noArray() );
      //printf("%f\n", dist[j]);
    }    
    max_value = max(dist);
    for(int i = 0; i < 8; i++)
    {
      if (max_value == dist[i])
      {
        rav4_2_array[i]++;
      }
    }
  }
  int rav4_1_array[8] = {0};
  for (int i = 0; i < s2.height; i++)
  {    
    for (int j = 0; j < 8; j++)
    {
      dist[j] = norm(descriptors_2.row(i), center.row(j), NORM_L2, cv::noArray() );
      //printf("%f\n", dist[j]);
    }    
    max_value = max(dist);
    for(int i = 0; i < 8; i++)
    {
      if (max_value == dist[i])
      {
        rav4_1_array[i]++;
      }
    }
  }
  int bus_array[8] = {0};
  for (int i = 0; i < s4.height; i++)
  {    
    for (int j = 0; j < 8; j++)
    {
      dist[j] = norm(descriptors_4.row(i), center.row(j), NORM_L2, cv::noArray() );
      //printf("%f\n", dist[j]);
    }    
    max_value = max(dist);
    for(int i = 0; i < 8; i++)
    {
      if (max_value == dist[i])
      {
        bus_array[i]++;
      }
    }
  }
  int jeep_array[8] = {0};
  for (int i = 0; i < s3.height; i++)
  {    
    for (int j = 0; j < 8; j++)
    {
      dist[j] = norm(descriptors_3.row(i), center.row(j), NORM_L2, cv::noArray() );
      //printf("%f\n", dist[j]);
    }    
    max_value = max(dist);
    for(int i = 0; i < 8; i++)
    {
      if (max_value == dist[i])
      {
        jeep_array[i]++;
      }
    }
  }
  Mat matched_image;
  double sum1 = 0;
  double sum2 = 0;
  double sum3 = 0;
  for (int i = 0; i < 8; i++)
  {sum1 = sum1 + pow(jeep_array[i] - rav4_2_array[i],2);}
  for (int i = 0; i < 8; i++)
  {sum2 = sum2 + pow(bus_array[i] - rav4_2_array[i],2);}
  for (int i = 0; i < 8; i++)
  {sum3 = sum3 + pow(rav4_1_array[i] - rav4_2_array[i],2);}

  cout << sum1 << endl<<sum2 <<endl<<sum3;
  //cout<< jeep_array <<endl;

  if (sum1 < sum2)
  {
    if(sum1 < sum3)
    {matched_image = jeep;}
    else
      {matched_image = rav4_1;}
  }
  else
  {
    if(sum2 < sum3)
    {matched_image = bus;}
    else
    {matched_image = rav4_1;}
  }
  
  imwrite("Matched_image_surf.png", matched_image);
  imshow("Matched_image_surf", matched_image);
/*
  BFMatcher matcher(NORM_L2  , true);
  vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );*/
  //std::cout << descriptors_group<<endl;
  //std::cout << center<<endl;
  //std::cout << descriptors_1.row(0);
  
/*
  drawKeypoints(rav4_1, keypoints_1, rav4_1_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints(rav4_2, keypoints_2, rav4_2_edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("keypoints_1",rav4_1_edges);
  imshow("keypoints_2",rav4_2_edges);*/
/*
  //vector<Dmatch>::const_iterator first = matches.begin();
  //vector<Dmatch>::const_iterator last = matches.begin() + 10;
  vector<DMatch> matches_count(matches.begin(), matches.end());	
  drawMatches(rav4_1, keypoints_1, rav4_2, keypoints_2, matches_count, rav_matches, Scalar::all(-1));	
  imshow("matches",rav_matches);
*/
    
  // cv::imshow("rav4_2_edges", rav4_2_edges);

  cv::waitKey(0); 
  return 0;
}

double max(double array[8])
{
  double temp;
  double array_sort[8];

  for (int i = 0; i < 8; i++)
  {array_sort[i] = array[i];}

  for (int i = 0; i < 7; i++)
  {
    for (int j = 0; j < 8; j++)
    {
      if (array_sort[j+1] > array_sort[j])
      {
        temp = array_sort[j];
        array_sort[j] = array_sort[j+1];
        array_sort[j+1] = temp;
      }
    }
  }
  return array_sort[0];

}