// Problem 1a - Created by Pranav Aggarwal - Texture Classification

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <sstream>
#include <string.h>
#include <sys/stat.h>


using namespace std;
void average(unsigned char [128][128][12], double [12]);
double max(double [12]);
double min(double [12]);
void print_images(cv::Mat, int );
cv::Mat Texture1_mat(128,128,CV_8U);
cv::Mat Texture2_mat(128,128,CV_8U);
cv::Mat Texture3_mat(128,128,CV_8U);
cv::Mat Texture4_mat(128,128,CV_8U);
cv::Mat Texture5_mat(128,128,CV_8U);
cv::Mat Texture6_mat(128,128,CV_8U);
cv::Mat Texture7_mat(128,128,CV_8U);
cv::Mat Texture8_mat(128,128,CV_8U);
cv::Mat Texture9_mat(128,128,CV_8U);
cv::Mat Texture10_mat(128,128,CV_8U);
cv::Mat Texture11_mat(128,128,CV_8U);
cv::Mat Texture12_mat(128,128,CV_8U);

class Texture
{
	double avg[12];
	double feature_space_data[12][25];
	
	public:
		//Texture();
		unsigned char Textures[128][128][12];

	public:
		void setTextures();
		void Normalize()
		{
			average(Textures, avg);
			for (int k = 0; k < 12; k++)
			{
				for (int i = 0 ; i < 128; i++)
				{
					for (int j = 0; j < 128; j++)
					{Textures[i][j][k] = Textures[i][j][k] - avg[k];}
				}
			}
		}
		void feature_extraction(double filter[5][5][25],double feature_space_data[][25]);
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
	int width = atoi(argv[1]);
	int height = atoi(argv[2]);

	///////////////////////////Problem 1a ////////////////////////////////
	Texture t;
	t.setTextures();
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
	double feature_space_data[12][25];
	t.feature_extraction(filter,feature_space_data);

	

	cv::Mat feature_space_data_cv(12,25,CV_32F);
	cv::Mat feature_space_data_cv_temp(12,1,CV_32F);

	for(int i = 0; i < 12; i++)
    {
        for(int j = 0; j < 25; j++)
        {
            feature_space_data_cv.at<float>(i,j) = feature_space_data[i][j];
        }
    }

  /*  for(int i = 0; i < 12; i++) //finding the best features.
    {
        feature_space_data_cv_temp.at<float>(i,0) = feature_space_data[i][0];        
    }*/

    // PCA using OpenCV
    cv::Mat reduced_feature_space(12,3,CV_32F);

    cv::PCA pca(feature_space_data_cv,cv::Mat(),CV_PCA_DATA_AS_ROW, 3); 
    pca.project(feature_space_data_cv,reduced_feature_space);

    //cout<<"PCA Projection Result:"<<endl;
    //cout<<reduced_feature_space<<endl;


    // k-means using OpenCV

    cv::Mat labels_full_dataset,center_full_dataset;
    double compactness_full_dataset;
    //cv::Mat centers(8, 1, CV_32FC1);
    cv::kmeans(feature_space_data_cv, 4, labels_full_dataset,
     		cv::TermCriteria(CV_TERMCRIT_ITER, 100, 1.0),
            100, cv::KMEANS_RANDOM_CENTERS,center_full_dataset);
    //printf("The output labels of the feature dataset with dim = 25: \n");
    //cout << labels_full_dataset<<endl;
    //cout << labels_full_dataset.at<int>(0)<<endl;
    //printf("The centorids are:\n");
    //cout << center_full_dataset;

   // char address[12];
    print_images(labels_full_dataset,1);

    double compactness = 0;
    double mean_variance = 0;
    /*for(int i = 0; i < 12; i++)
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
    }*/
    //printf("Compactness is %f and the spreading of the clusters is %f\n", compactness,mean_variance);

    double centroids[4][3] = {{-2,0,-0.2},{-2,0.5,-0.5},{-2,-0.5,0.8},{-2,1.5,0.8}};
	cv::Mat labels_pca_dataset,center_pca_dataset;
	double compactness_pca_dataset;
    //cv::Mat centers(8, 1, CV_32FC1);
    cv::kmeans(reduced_feature_space, 4, labels_pca_dataset,
            cv::TermCriteria(CV_TERMCRIT_ITER, 100, 1.0),
            100, cv::KMEANS_RANDOM_CENTERS,center_pca_dataset);//{{-2,0,-0.2},{-2,0.5,-0.5},{-2,-0.5,0.8},{-2,1.5,0.8}}
    //printf("\nThe output labels of the feature dataset with dim = 3: \n");
    //cout << labels_pca_dataset;
    //printf("The centorids are:\n");
    //cout << center_pca_dataset;

    print_images(labels_pca_dataset,2);





	/*if (!(file=fopen("Warped_Image.raw","wb"))) {
		cout << "Cannot open file: " << "Warped_Image.raw" << endl;
		exit(1);
	}
	fwrite(ImagedataWarped, sizeof(unsigned char), row*col*BytesPerPixel, file);
	fclose(file);	*/

	return 0;
}

/*Texture::Texture(int const width1, int const height1)
{
	width = width1;
	height = height1;
}*/

void Texture::feature_extraction(double filter[5][5][25],double feature_space_data[12][25])
{
	double sum = 0;
	double sum_mask = 0;
	double avg_feature[12][25] = {0};

	for(int filter_number = 0; filter_number < 25; filter_number++)
	{
		for (int k = 0; k < 12; k++)
		{
			sum = 0;
			for (int i = 0 ; i < 128; i++)
			{
				for (int j = 0; j < 128; j++)
				{
					sum_mask = 0;
					for(int mask_row = -2; mask_row < 3; mask_row++)
					{
						for (int mask_col = -2; mask_col < 3; mask_col++)
						{
							if (i + mask_row > 0 && i + mask_row < 128 && j + mask_col > 0 && j + mask_col < 128 )
							{
								sum_mask = sum_mask + Textures[i + mask_row][j+mask_col][k] * filter[mask_row + 2][mask_col + 2][filter_number];
							}
							else
							{
								sum_mask = sum_mask + Textures[i][j][k] * filter[mask_row + 2][mask_col + 2][filter_number];
							}							
						}
					}
					sum = sum + abs(sum_mask);
				}
			}
			avg_feature[k][filter_number] = sum / (128.0 * 128.0);
			//printf("%f  ", avg_feature[k][filter_number]);
		}
		//printf("\n");
	}

	double array_temp[12];
	double feature_min, feature_max;
	
	for (int feature = 0 ; feature < 25; feature++)
	{
		// finding max and min
		for (int sample = 0; sample < 12; sample++)
		{
			array_temp[sample] = avg_feature[sample][feature];
			//printf("%f\n", array_temp[sample]);
		}
		feature_max = max(array_temp);
		feature_min = min(array_temp);
		//printf("\n%f\n", feature_max);
		//printf("\n%f\n", feature_min);

		// preprocessing of features
		for (int sample = 0; sample < 12; sample++)
		{
			feature_space_data[sample][feature] = (avg_feature[sample][feature] - feature_min)/(feature_max - feature_min);
			//printf("%f   ", feature_space_data[sample][feature]);
		}
		//printf("\n");
	}

}
void average(unsigned char Textures_temp[128][128][12],double avg_temp[12])
{
	double sum[12] = {0};
	for (int k = 0; k < 12; k++)
	{
		for (int i = 0 ; i < 128; i++)
		{
			for (int j = 0; j < 128; j++)
			{				
				sum[k] = sum[k] + Textures_temp[i][j][k];
			}
		}
		avg_temp[k] = sum[k]/(128*128);
		//printf("%f\n", avg_temp[k]);
	}
}
void Texture::setTextures()
{
	FILE *file;
	unsigned char Texture1[128][128];
	unsigned char Texture2[128][128];
	unsigned char Texture3[128][128];
	unsigned char Texture4[128][128];
	unsigned char Texture5[128][128];
	unsigned char Texture6[128][128];
	unsigned char Texture7[128][128];
	unsigned char Texture8[128][128];
	unsigned char Texture9[128][128];
	unsigned char Texture10[128][128];
	unsigned char Texture11[128][128];
	unsigned char Texture12[128][128];	
	if (!(file=fopen("Texture1.raw","rb"))) {
	cout << "Cannot open file: " << "Texture1.raw" <<endl;
	exit(1);
	}
	fread(Texture1, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture1_mat = cv::Mat(128, 128, CV_8U, Texture1);

	if (!(file=fopen("Texture2.raw","rb"))) {
	cout << "Cannot open file: " << "Texture2.raw" <<endl;
	exit(1);
	}
	fread(Texture2, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture2_mat = cv::Mat(128, 128, CV_8U, Texture2);

	if (!(file=fopen("Texture3.raw","rb"))) {
	cout << "Cannot open file: " << "Texture3.raw" <<endl;
	exit(1);
	}
	fread(Texture3, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture3_mat = cv::Mat(128, 128, CV_8U, Texture3);

	if (!(file=fopen("Texture4.raw","rb"))) {
	cout << "Cannot open file: " << "Texture4.raw" <<endl;
	exit(1);
	}
	fread(Texture4, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture4_mat = cv::Mat(128, 128, CV_8U, Texture4);

	if (!(file=fopen("Texture5.raw","rb"))) {
	cout << "Cannot open file: " << "Texture5.raw" <<endl;
	exit(1);
	}
	fread(Texture5, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture5_mat = cv::Mat(128, 128, CV_8U, Texture5);

	if (!(file=fopen("Texture6.raw","rb"))) {
	cout << "Cannot open file: " << "Texture6.raw" <<endl;
	exit(1);
	}
	fread(Texture6, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture6_mat = cv::Mat(128, 128, CV_8U, Texture6);

	if (!(file=fopen("Texture7.raw","rb"))) {
	cout << "Cannot open file: " << "Texture7.raw" <<endl;
	exit(1);
	}
	fread(Texture7, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture7_mat = cv::Mat(128, 128, CV_8U, Texture7);

	if (!(file=fopen("Texture8.raw","rb"))) {
	cout << "Cannot open file: " << "Texture8.raw" <<endl;
	exit(1);
	}
	fread(Texture8, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture8_mat = cv::Mat(128, 128, CV_8U, Texture8);

	if (!(file=fopen("Texture9.raw","rb"))) {
	cout << "Cannot open file: " << "Texture9.raw" <<endl;
	exit(1);
	}
	fread(Texture9, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture9_mat = cv::Mat(128, 128, CV_8U, Texture9);

	if (!(file=fopen("Texture10.raw","rb"))) {
	cout << "Cannot open file: " << "Texture10.raw" <<endl;
	exit(1);
	}
	fread(Texture10, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture10_mat = cv::Mat(128, 128, CV_8U, Texture10);

	if (!(file=fopen("Texture11.raw","rb"))) {
	cout << "Cannot open file: " << "Texture11.raw" <<endl;
	exit(1);
	}
	fread(Texture11, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture11_mat = cv::Mat(128, 128, CV_8U, Texture11);

	if (!(file=fopen("Texture12.raw","rb"))) {
	cout << "Cannot open file: " << "Texture12.raw" <<endl;
	exit(1);
	}
	fread(Texture12, sizeof(unsigned char), 128*128, file);
	fclose(file);
	Texture12_mat = cv::Mat(128, 128, CV_8U, Texture12);

	for(int i = 0; i < 128; i++)
	{
		for (int j = 0; j < 128; j++)
		{
			Textures[i][j][0] = Texture1[i][j];
			Textures[i][j][1] = Texture2[i][j];
			Textures[i][j][2] = Texture3[i][j];
			Textures[i][j][3] = Texture4[i][j];
			Textures[i][j][4] = Texture5[i][j];
			Textures[i][j][5] = Texture6[i][j];
			Textures[i][j][6] = Texture7[i][j];
			Textures[i][j][7] = Texture8[i][j];
			Textures[i][j][8] = Texture9[i][j];
			Textures[i][j][9] = Texture10[i][j];
			Textures[i][j][10] = Texture11[i][j];
			Textures[i][j][11] = Texture12[i][j];
		}
	}
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

void print_images(cv::Mat labels, int count)
{
	//mkdir("c:/myfolder");
	int number = 1;
	stringstream dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8,dir9,dir10,dir11,dir12;
	//dir13 << "Images"<<count<< "/Type1";
	//dir14 << "Images"<<count<< "/Type2";
	//dir15 << "Images"<<count<< "/Type3";
	//dir16 << "Images"<<count<< "/Type4";
	//cout << dir13;
	//mkdir((dir13.str()).c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	//cout << dir1;

	//mkdir((dir14.str()).c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	//mkdir((dir15.str()).c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	//mkdir((dir16.str()).c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    dir1<<"Images"<<count<<"/Type" << labels.at<int>(0) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir1.str(),Texture1_mat); number++;
    dir2<<"Images"<<count<<"/Type" << labels.at<int>(1) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir2.str(),Texture2_mat);number++;
    dir3<<"Images"<<count<<"/Type" << labels.at<int>(2) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir3.str(),Texture3_mat);number++;
    dir4<<"Images"<<count<<"/Type" << labels.at<int>(3) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir4.str(),Texture4_mat);number++;
    dir5<<"Images"<<count<<"/Type" << labels.at<int>(4) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir5.str(),Texture5_mat);number++;
    dir6<<"Images"<<count<<"/Type" << labels.at<int>(5) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir6.str(),Texture6_mat);number++;
    dir7<<"Images"<<count<<"/Type" << labels.at<int>(6 + 1)<<"/Texture"<<number<<".png";
    cv::imwrite(dir7.str(),Texture7_mat);number++;
    dir8<<"Images"<<count<<"/Type" << labels.at<int>(7) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir8.str(),Texture8_mat);number++;
    dir9<<"Images"<<count<<"/Type" << labels.at<int>(8) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir9.str(),Texture9_mat);number++;
    dir10<<"Images"<<count<<"/Type" << labels.at<int>(9) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir10.str(),Texture10_mat);number++;
    dir11<<"Images"<<count<<"/Type" << labels.at<int>(10) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir11.str(),Texture11_mat);number++;
    dir12<<"Images"<<count<<"/Type" << labels.at<int>(11) + 1<<"/Texture"<<number<<".png";
    cv::imwrite(dir12.str(),Texture12_mat);
    /*dir1<<"Images1/Type1/Texture1.jpg";
    cv::imwrite(dir1.str(),Texture1_mat); number++;*/
}