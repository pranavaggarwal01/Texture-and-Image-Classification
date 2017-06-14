// Problem 1b - Created by Pranav Aggarwal - Texture Segmentation

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <sstream>

using namespace std;
double average(unsigned char [512][512],int , int );
double max(double [12]);
double min(double [12]);

double feature_space_data[512 * 512][25] = {0};
double Texture_filtered[512][512][25] = {0};

class Texture
{
	double avg;
	double feature_space_data[12][25];
	
	public:
		//Texture();
		unsigned char Texture[512][512];
		int width;
		int height;

	public:
		void setTextures(string file_name, int row, int col);
		void Normalize()
		{
			avg = average(Texture, width, height);
			
			for (int i = 0 ; i < 512; i++)
			{
				for (int j = 0; j < 512; j++)
				{Texture[i][j] = Texture[i][j] - avg;}
			}
			
		}
		void feature_extraction(double filter[5][5][25],double feature_space_data[512 * 512][25]);
};



int main(int argc, char *argv[])

{
	// Define file pointer and variables
	
	int BytesPerPixel = 1;
	int Size = 256;
	
	// Check for proper syntax
	if (argc < 2){
		cout << "Syntax Error - Incorrect Parameter Usage:" << endl;
		cout << "program_name width height" << endl;
		return 0;
	}
	string file_name = argv[1];	
	int width = atoi(argv[2]);
	int height = atoi(argv[3]);

	///////////////////////////Problem 1a ////////////////////////////////
	Texture t;
	t.setTextures(file_name, height, width);
	t.Normalize();
	double filter[5][5][25] = {0};
	double laws_filters[5][5] = {{1,4,6,4,1},
				{-1,-2,0,2,1},
				{-1,0,2,0,-1},
				{-1,2,0,-2,1},
				{1,-4,6,-4,1}};

	for (int c = 0; c < 5; c++)
	{
		for(int k = 0; k < 5; k++) // filterwise
		{			
			for(int i = 0; i < 5; i++) // filterwise
			{
				for (int j = 0; j < 5; j++)
				{
					filter[i][j][(c*5)+k] = laws_filters[c][j] * laws_filters[k][i];
					//printf("%f    ", filter[i][j][(c*5)+k]);	
				}
				//printf("\n");
			} 
			//printf("\n\n");
		}
		//printf("\n\n");
	}
	
	t.feature_extraction(filter,feature_space_data);

	

	cv::Mat feature_space_data_cv(t.height * t.width,25,CV_32F);

	for(int i = 0; i < t.height*t.width; i++)
    {
        for(int j = 0; j < 25; j++)
        {
            feature_space_data_cv.at<float>(i,j) = feature_space_data[i][j];
        }
    }

    // PCA using OpenCV
    cv::Mat reduced_feature_space(t.height*t.width,3,CV_32F);

    cv::PCA pca(feature_space_data_cv,cv::Mat(),CV_PCA_DATA_AS_ROW, 3); 
    pca.project(feature_space_data_cv,reduced_feature_space);

    //cout<<"PCA Projection Result:"<<endl;
    //cout<<reduced_feature_space<<endl;


    // k-means using OpenCV
    int no_of_clutters;
    cout << "Clutter size:\n";
    cin >> no_of_clutters;	
    
    cv::Mat labels_full_dataset,center_full_dataset;
    double compactness_full_dataset;
    //cv::Mat centers(8, 1, CV_32FC1);
    cv::kmeans(feature_space_data_cv, no_of_clutters, labels_full_dataset,
     		cv::TermCriteria(CV_TERMCRIT_ITER, 10, 1.0),
            1000, cv::KMEANS_RANDOM_CENTERS,center_full_dataset);
    //cout << labels_full_dataset<<endl;
    //cout << labels_full_dataset.at<int>(0)<<endl;
    //cout << center_full_dataset;
    cv::Mat segmentedTexture(height,width,CV_8U);
    for (int i = 0; i < height; i++)
    {
    	for (int j = 0; j < width; j++)
    	{
    		segmentedTexture.at<unsigned char>(i,j) = int (labels_full_dataset.at<int>((i * width) + j) * (255 / (no_of_clutters-1)));
    	}
    }
    imwrite("Segmented_Texture_all_features1.png", segmentedTexture);


    /*double compactness = 0;
    double mean_variance = 0;
    for(int i = 0; i < 12; i++)
    {
    	for(int j = 0; j < 25; j++)
    	{
    		compactness = compactness + (pow(feature_space_data_cv.at<float>(i,j) - 
    			center_full_dataset.at<float>(labels_full_dataset.at<int>(i),j),2));
    	}
    }

    for (int i = 0; i < 3; i ++)
    {
    	for(int k = i+1; k < 4; k++)
    	{
    		for(int j = 0; j < 25; j++)
			{
				mean_variance = mean_variance + pow(center_full_dataset.at<float>(i,j) - 
					center_full_dataset.at<float>(k,j),2);
			}
    	}
    }
    //printf("Compactness is %f and the spreading of the clusters is %f\n", compactness,mean_variance);
*/
	cv::Mat labels_pca_dataset,center_pca_dataset;
	double compactness_pca_dataset;
    //cv::Mat centers(8, 1, CV_32FC1);
    cv::kmeans(reduced_feature_space, no_of_clutters, labels_pca_dataset,
            cv::TermCriteria(CV_TERMCRIT_ITER, 10, 1.0),
            1000, cv::KMEANS_RANDOM_CENTERS,center_pca_dataset);
    //cout << labels_pca_dataset;
    //cout << center_pca_dataset;
    for (int i = 0; i < height; i++)
    {
    	for (int j = 0; j < width; j++)
    	{
    		segmentedTexture.at<unsigned char>(i,j) = int (labels_pca_dataset.at<int>((i * width) + j) * (255 / (no_of_clutters-1)));
    	}
    }
    imwrite("Segmented_Texture_reduced_features1.png", segmentedTexture);

	return 0;
}



void Texture::setTextures(string file_name, int row, int col)
{
	unsigned char Texture1[row][col];
	FILE *file;
	const char* file_name1 = file_name.c_str();
	if (!(file=fopen(file_name1,"rb"))) {
	cout << "Cannot open file: " << file_name1 <<endl;
	exit(1);
	}
	fread(Texture1, sizeof(unsigned char), row*col, file);
	fclose(file);
	for(int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{Texture[i][j] = Texture1[i][j];}
	}
	width = col;
	height = row;
	
}

void Texture::feature_extraction(double filter[5][5][25],double feature_space_data[512*512][25])
{

	double sum = 0;
	double sum_mask = 0;
	//double avg_feature[12][25] = {0};
	
	int count;
	for(int filter_number = 0; filter_number < 25; filter_number++)
	{		
		for (int i = 0 ; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				sum_mask = 0;
				count = 0;
				
				for(int mask_row = -2; mask_row < 3; mask_row++)
				{
					for (int mask_col = -2; mask_col < 3; mask_col++)
					{
						if (i + mask_row > 0 && i + mask_row < height && j + mask_col > 0 && j + mask_col < width )
						{
							sum_mask = sum_mask + Texture[i + mask_row][j+mask_col] * filter[mask_row + 2][mask_col + 2][filter_number];
							count++;
						}
					}					
				}
				Texture_filtered[i][j][filter_number] = abs(sum_mask)/(count);				
			}
		}
	}
	//printing the filtered images
	double image_temp[height][width];
	cv::Mat image_mat,filtered_mat;
	double min_val, max_val;
	image_mat.create(height,width,CV_32FC2);
	filtered_mat.create(height,width,CV_8U);
	
	int len;
	for(int filter_number = 0; filter_number < 25; filter_number++)
	{		
		for (int i = 0 ; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				image_mat.at<float>(i,j) = Texture_filtered[i][j][filter_number];
			}
		}
		
		cv::minMaxLoc(image_mat, &min_val, &max_val);

		for (int i = 0 ; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				filtered_mat.at<unsigned char>(i,j) = int(((image_mat.at<float>(i,j) - min_val)/(max_val - min_val)) * 255);
			}
		}
		stringstream s;
    	s << "filtered_images/filtered_image"<<filter_number+1<<".png";
    	//cout << s;
		cv::imwrite(s.str(),filtered_mat);
		//delete s;

	}
	cout << "\nEnter mask_size:\n";
	int mask_size;
	cin >> mask_size;
	for(int filter_number = 0; filter_number < 25; filter_number++)
	{		
		for (int i = 0 ; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				sum_mask = 0;
				count = 0;
				
				for(int mask_row = -(mask_size-1)/2; mask_row <= (mask_size-1)/2; mask_row++)
				{
					for (int mask_col = -(mask_size-1)/2; mask_col <= (mask_size-1)/2; mask_col++)
					{
						if (i + mask_row > 0 && i + mask_row < height && j + mask_col > 0 && j + mask_col < width )
						{
							sum_mask = sum_mask + Texture_filtered[i + mask_row][j+mask_col][filter_number];
							count++;
						}
					}					
				}
				feature_space_data[(i * width) + j][filter_number] = abs(sum_mask)/(count);
			}
		}
	}

	/*double array_temp[12];
	double feature_min, feature_max;
	
	for (int feature = 0 ; feature < 25; feature++)
	{
		// finding max and min
		for (int sample = 0; sample < 12; sample++)
		{
			array_temp[sample] = avg_feature[sample][feature];
			printf("%f\n", array_temp[sample]);
		}
		feature_max = max(array_temp);
		feature_min = min(array_temp);
		printf("\n%f\n", feature_max);
		printf("\n%f\n", feature_min);

		// preprocessing of features
		for (int sample = 0; sample < 12; sample++)
		{
			feature_space_data[sample][feature] = (avg_feature[sample][feature] - feature_min)/(feature_max - feature_min);
			printf("%f   ", feature_space_data[sample][feature]);
		}
		printf("\n");
	}*/

}
double average(unsigned char Texture_temp[512][512],int width, int height)
{
	double sum = 0;
	double avg_temp;
	
	for (int i = 0 ; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{				
			sum = sum + Texture_temp[i][j];
		}
	}
	avg_temp = sum / (height * width);
	//printf("%f\n", avg_temp);
	return avg_temp;	
}


double max(double array[12])
{
	double temp;
	double array_sort[12];

	for (int i = 0; i < 12; i++)
	{array_sort[i] = array[i];}

	for (int i = 0; i < 11; i++)
	{
		for (int j = 0; j < 12; j++)
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

double min(double array[12])
{
	double temp ;
	double array_sort[12];

	for (int i = 0; i < 12; i++)
	{array_sort[i] = array[i];}

	for (int i = 0; i < 11; i++)
	{
		for (int j = 0; j < 12; j++)
		{
			if (array_sort[j+1] < array_sort[j])
			{
				temp = array_sort[j];
				array_sort[j] = array_sort[j+1];
				array_sort[j+1] = temp;

			}
		}
	}
	//printf("%f\n", array_sort[0]);
	return array_sort[0];

}