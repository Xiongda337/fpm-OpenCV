/*
fpmMain.cpp
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
//#include <list>
#include<vector>
//#include "include/rapidjson"
#include "domeHoleCoordinates.h"

using namespace std;
using namespace cv;

#define FILENAME_LENGTH 128
#define FILE_HOLENUM_DIGITS 4

string filePrefix = "iLED_";

class FPMimg{
  public:
        cv::Mat Image;
        int led_num;
        float sinTheta_x;
        float sinTheta_y;
};

class FPM_Dataset{
  public:
        string datasetRoot;
        std::vector<FPMimg> imageStack;
        uint16_t ledCount;
        float pixelSize;         // pixel size in microns
        float objectiveMag;
        float objectiveNA;
        float lambda;              // wavelength in microns
        bool color;                // flag for data acquired on color camera
        int16_t centerLED = 249;
        int16_t cropX;
        int16_t cropY;
        int16_t Np;
};

/*/Check if file exists
inline bool exists_test (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}*/

int loadDataset(FPM_Dataset *dataset) {
	DIR *dir;
	struct dirent *ent;
	Mat tmpImg;
	if ((dir = opendir (dataset->datasetRoot.c_str())) != NULL) {
      int num_images = 0;
      cout << "Loading Images..." << endl;
	    while ((ent = readdir (dir)) != NULL) {
		  //add ent to list
    		  string fileName = ent->d_name;
          /* Get data from file name, if name is right format.*/
    		  if (fileName.compare(".") != 0 && fileName.compare("..") != 0 && (strcmp (".tif", &(ent->d_name[strlen( ent->d_name) - 4])) == 0)) {
    		      string holeNum = fileName.substr(fileName.find(filePrefix)+filePrefix.length(),FILE_HOLENUM_DIGITS);
               FPMimg currentImage;
    		      currentImage.led_num = atoi(holeNum.c_str());
    		      tmpImg = imread(dataset->datasetRoot + "/" + fileName, -1);
    		      currentImage.Image = tmpImg(cv::Rect(dataset->cropX,dataset->cropY,dataset->Np,dataset->Np)).clone();
          		currentImage.sinTheta_x = sin(atan(domeHoleCoordinates[currentImage.led_num][0] / domeHoleCoordinates[currentImage.led_num][2]));
          		currentImage.sinTheta_y = sin(atan(domeHoleCoordinates[currentImage.led_num][1] / domeHoleCoordinates[currentImage.led_num][2]));
          		dataset->imageStack.push_back(currentImage);
          		num_images++;
          		std::cout << "Loaded: " << fileName << ", LED # is: " << currentImage.led_num << std::endl;
          		std::cout << "   theta_x is: " << currentImage.sinTheta_x << ", theta_y is: " << currentImage.sinTheta_y <<std::endl;
          		std::cout << "Image Size is: " << currentImage.Image.rows << " x " << currentImage.Image.cols << endl;
    		  }
	    }
	  closedir (dir);
	  return num_images;

	} else {
	  /* could not open directory */
	  std::cout << "ERROR: Could not Load Images.\n";
	  //perror ("");
	  return EXIT_FAILURE;
	}
}

cv::Mat dtf2(Mat inImg, Mat outImg)
{
   Mat padded;
   int m = getOptimalDFTSize(inImg.rows);
   int n = getOptimalDFTSize(inImg.cols);
   copyMakeBorder(inImg, padded, 0, m - inImg.rows, 0, n - inImg.cols, BORDER_CONSTANT, Scalar::all(0));
   
   Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
   Mat complexI;
   merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

   dft(complexI, complexI);            // this way the result may fit in the source matrix

   // FFT Shift the result
   int cx = complexI.cols/2;
   int cy = complexI.rows/2;

   Mat q0(complexI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
   Mat q1(complexI, Rect(cx, 0, cx, cy));  // Top-Right
   Mat q2(complexI, Rect(0, cy, cx, cy));  // Bottom-Left
   Mat q3(complexI, Rect(cx, cy, cx, cy)); // Bottom-Right
    //split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
     Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
   q0.copyTo(tmp);
   q3.copyTo(q0);
   tmp.copyTo(q3);

   q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
   q2.copyTo(q1);
   tmp.copyTo(q2);
}

cv::Mat idft2(Mat inImg, Mat outImg)
{
    //calculating the idft
    cv::Mat inverseTransform;
    cv::dft(inImg, outImg, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    //normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
   // imshow("Reconstructed", inverseTransform);
    //waitKey();
}

void showImg(Mat img)
{
   //normalize(img, img, 0, 1, CV_MINMAX);
   namedWindow("Img", WINDOW_NORMAL);
   imshow("Img", img);
   waitKey();
}

int main(int argc, char** argv )
{
   // Parameters from the .m file
   cout << "working!" <<endl;
   
   FPM_Dataset mDataset;
   mDataset.datasetRoot = "/home/zfphil/Dropbox/FP_mono_nofilter/";
   mDataset.ledCount = 508;
   mDataset.pixelSize = 6.5;
   mDataset.objectiveMag = 40*2;
   mDataset.objectiveNA = 0.2;
   mDataset.color = false;
   mDataset.centerLED = 249;
   
   char fileName[FILENAME_LENGTH];
   sprintf(fileName,"%s%04d.tif",filePrefix.c_str(),mDataset.centerLED);
   cout << mDataset.datasetRoot + fileName <<endl;

   Mat centerImg = imread(mDataset.datasetRoot + fileName, -1);
   
   uint16_t Np = 90;
   int16_t cropX = 1248;
   int16_t cropY = 1020;
   cv::Mat croppedRef(centerImg,cv::Rect(cropX,cropY,Np,Np));
   showImg(croppedRef);
   
   mDataset.Np = Np;
   mDataset.cropX = cropX;
   mDataset.cropY = cropY;
   loadDataset(&mDataset);
   
   
}
