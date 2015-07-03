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
#include <fstream>
//#include <list>
#include<vector>
//#include "include/rapidjson"
#include "domeHoleCoordinates.h"

using namespace std;
using namespace cv;

#define FILENAME_LENGTH 128
#define FILE_HOLENUM_DIGITS 4

string filePrefix = "iLED_";

bool preprocessDebug = false;

class FPMimg{
  public:
        cv::Mat Image;
        int led_num;
        float sinTheta_x;
        float sinTheta_y;
        float vled;
        float uled;
        int16_t idx_u;
        int16_t idx_v;
        float illumination_na;
        int16_t bg_val;
};

class FPM_Dataset{
  public:
        string                datasetRoot;
        std::vector<FPMimg>   imageStack;
        uint16_t              ledCount;
        float                 pixelSize;           // pixel size in microns
        float                 objectiveMag;
        float                 objectiveNA;
        float                 lambda;              // wavelength in microns
        bool                  color;                // flag for data acquired on color camera
        int16_t               centerLED = 249;     // Closest LED to center of Array
        int16_t               cropX;
        int16_t               cropY;
        int16_t               Np;
        float                 du;                // Delta Spatial Frequency
        std::vector<float>    NALedPatternStackX;
        std::vector<float>    NALedPatternStackY;
        std::vector<float>    illuminationNAList;
        std::vector<float>    sortedNALedPatternStackX;
        std::vector<float>    sortedNALedPatternStackY;
        std::vector<int16_t>  sortedIndicies;
        int16_t               bk1cropX;
        int16_t               bk1cropY;     
        int16_t               bk2cropX;     
        int16_t               bk2cropY;
        float                 bgThreshold;
        float                 ps_eff;
};

/*/Check if file exists
inline bool exists_test (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}*/

void showImg(Mat img)
{
   normalize(img, img, 0, 1, CV_MINMAX);
   namedWindow("Img", WINDOW_NORMAL);
   imshow("Img", img);
   waitKey();
}

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


int loadDataset(FPM_Dataset *dataset) {
	DIR *dir;
	struct dirent *ent;
	Mat fullImg;
	FPMimg tmpFPMimg;
	tmpFPMimg.Image = Mat::zeros(dataset->Np, dataset->Np, CV_8UC1);
	
	// Initialize array of image objects, since we don't know exactly what order imread will read in images. First (0th) element is a dummy so we can access these using the led # directly
	for (int16_t ledIdx = 0; ledIdx <= dataset->ledCount; ledIdx++)
	{

	    dataset->imageStack.push_back(tmpFPMimg);
       dataset->illuminationNAList.push_back(-1.0);
       dataset->NALedPatternStackX.push_back(-1.0);
       dataset->NALedPatternStackY.push_back(-1.0);
	}

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
    		      fullImg = imread(dataset->datasetRoot + "/" + fileName, CV_LOAD_IMAGE_ANYDEPTH);
               
    		      currentImage.Image = fullImg(cv::Rect(dataset->cropX,dataset->cropY,dataset->Np,dataset->Np)).clone();
          		currentImage.sinTheta_x = sin(atan2(domeHoleCoordinates[currentImage.led_num-1][1], domeHoleCoordinates[currentImage.led_num-1][2]));
          		currentImage.sinTheta_y = sin(atan2(domeHoleCoordinates[currentImage.led_num-1][0] , domeHoleCoordinates[currentImage.led_num-1][2]));
          		currentImage.illumination_na = sqrt(currentImage.sinTheta_x*currentImage.sinTheta_x+currentImage.sinTheta_y*currentImage.sinTheta_y);
          		
          		cv::Scalar bk1 = cv::mean(fullImg(cv::Rect(dataset->bk1cropX,dataset->bk1cropY,dataset->Np,dataset->Np)));
               cv::Scalar bk2 = cv::mean(fullImg(cv::Rect(dataset->bk2cropX,dataset->bk2cropY,dataset->Np,dataset->Np)));
               
               double bg_val = ((double)bk2[0]+(double)bk1[0])/2;
               if (bg_val > dataset->bgThreshold)
                  bg_val = dataset->bgThreshold;
               
               currentImage.bg_val = (int16_t)round(bg_val);
               
               // Perform Background Subtraction
               cv::subtract(currentImage.Image,cv::Scalar(currentImage.bg_val,0,0),currentImage.Image);
               //cv::subtract(currentImage.Image,bk,currentImage.Image);
          		
          		currentImage.uled = currentImage.sinTheta_x/dataset->lambda;
          		currentImage.vled = currentImage.sinTheta_y/dataset->lambda;
          		
               currentImage.idx_u = (int16_t) round(currentImage.uled/dataset->du);
          		currentImage.idx_v = (int16_t) round(currentImage.vled/dataset->du);
          		
          		dataset->imageStack.at(currentImage.led_num) = currentImage;
          		
               // Need to ensure we do a deep copy of the image?
          		//dataset->imageStack.at(currentImage.led_num).Image = currentImage.Image.clone();
          		
          		dataset->illuminationNAList.at(currentImage.led_num) = currentImage.illumination_na;
          		dataset->NALedPatternStackX.at(currentImage.led_num) = currentImage.sinTheta_x;
          		dataset->NALedPatternStackY.at(currentImage.led_num) = currentImage.sinTheta_y;
          		
          		num_images++;
          		std::cout << "Loaded: " << fileName << ", LED # is: " << currentImage.led_num << std::endl;
               if (preprocessDebug)
               {
             		std::cout << "   sintheta_x is: " << currentImage.sinTheta_x << ", sintheta_y is: " << currentImage.sinTheta_y <<std::endl;
             		std::cout << "   Image Size is: " << currentImage.Image.rows << " x " << currentImage.Image.cols << endl;
             		cout << "   cartx is: " << domeHoleCoordinates[currentImage.led_num-1][0] << endl;
             		cout << "   carty is: " << domeHoleCoordinates[currentImage.led_num-1][1] << endl;
             		cout << "   cartz is: " << domeHoleCoordinates[currentImage.led_num-1][2] << endl;
             		cout << "   atan(cartx) : " << atan(domeHoleCoordinates[currentImage.led_num-1][0] / domeHoleCoordinates[currentImage.led_num-1][2]) <<endl;
             		cout << "   atan(carty) : " << atan(domeHoleCoordinates[currentImage.led_num-1][1] / domeHoleCoordinates[currentImage.led_num-1][2]) <<endl;
             		
             	   cout << "   sin(atan(cartx)) : " << sin(atan(domeHoleCoordinates[currentImage.led_num-1][0] / domeHoleCoordinates[currentImage.led_num-1][2])) <<endl;
             		cout << "   sin(atan(carty)) : " << sin(atan(domeHoleCoordinates[currentImage.led_num-1][1] / domeHoleCoordinates[currentImage.led_num-1][2])) <<endl;
             		
             		cout << "   uled : " << currentImage.uled <<endl;
             		cout << "   vled : " << currentImage.vled <<endl;
             		
             		cout << "   idx_u : " << currentImage.idx_u <<endl;
             		cout << "   idx_v : " << currentImage.idx_v <<endl;

                  cout << "   illumination na: " << currentImage.illumination_na <<std::endl;
                  cout << std::endl<< std::endl;
               }
               
    		  }
	    }
	  closedir (dir);
	  
	  // Sort the Images into the correct order
	  int16_t indexIncr = 1;
	  for (auto i: sort_indexes(dataset->illuminationNAList)) {
	      //cout << dataset->illuminationNAList.at(i) <<endl;
	      //cout << i <<endl;
         //cout << dataset->NALedPatternStack[i] << endl;
         dataset->sortedIndicies.push_back(i);
         dataset->sortedNALedPatternStackX.push_back(dataset->NALedPatternStackX[i]);
         dataset->sortedNALedPatternStackY.push_back(dataset->NALedPatternStackY[i]);
         //cout << dataset->sortedNALedPatternStackX.at(indexIncr)<< endl;
         indexIncr++;
         //cout << " led "<< indexIncr <<" which has na of "<< dataset->illuminationNAList.at(indexIncr)<<" is sorted as " << i <<endl;
      }
      /*
      for (int led=1; led<=508; led++)
      {
      
         cout << "LED# " << led << " With NA of "<< dataset->illuminationNAList.at(led) << " is ordered at: " << dataset->sortedIndicies.at(led) << " with a bg val of " <<dataset->imageStack.at(led).bg_val << endl;
      }
      */
	  return num_images;

	} else {
	  /* could not open directory */
	  std::cout << "ERROR: Could not Load Images.\n";
	  //perror ("");
	  return EXIT_FAILURE;
	}
}

/*
void fpmBackgroundSubtraction(FPM_Dataset *dataset)
{
   for (int16_t imgIdx=0; imgIdx < dataset->ledCount; imgIdx++)
   {

   }
}
*/


void saveFullDataset (FPM_Dataset *dataset, string path)
{
   std::cout << "Saving current dataset to: " + path << std::endl;
   string fname1 = path + "fpmDataset_LEDs" + ".csv";
   ofstream fileObj (fname1.c_str());
   //fileObj << "FORMAT: int led_num, float sinTheta_x, float sinTheta_y, float vled, float uled,int16_t idx_u, int16_t idx_v, float illumination_na, int16_t bg_val;"<<std::endl;
   string fname2 = path + "fpmDataset_globalVars" + ".csv";
   ofstream fileObj2 (fname2.c_str());
   // Save Images and led parameters
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      imwrite(path + "fpmDataset_LED" + std::to_string(dataset->imageStack.at(imgIdx).led_num) + ".tif",dataset->imageStack.at(imgIdx).Image);
      fileObj << dataset->imageStack.at(imgIdx).led_num << "," << dataset->imageStack.at(imgIdx).sinTheta_x << ", " << dataset->imageStack.at(imgIdx).sinTheta_y << ", " << dataset->imageStack.at(imgIdx).vled << ", " << dataset->imageStack.at(imgIdx).uled << ", " << dataset->imageStack.at(imgIdx).idx_u << ", " << dataset->imageStack.at(imgIdx).idx_v << ", " << dataset->imageStack.at(imgIdx).illumination_na << ", " << dataset->imageStack.at(imgIdx).bg_val << ";" << std::endl;
   }
   fileObj.close();
   
   
   //fileObj2 << "FORMAT: string datasetRoot, uint16_t ledCount, float pixelSize, float objectiveMag, float objectiveNA, float lambda, bool color, int16_t centerLED, int16_t cropX, int16_t cropY, int16_t Np, float du, int16_t bk1cropX, int16_t bk1cropY, int16_t bk2cropX, int16_t bk2cropY, float bgThreshold;"<< std::endl;
   
   fileObj2 << dataset->datasetRoot << "," << dataset->ledCount << "," << dataset->pixelSize << "," << dataset->objectiveMag << "," << dataset->objectiveNA << "," << dataset->lambda << "," << dataset->color << "," << dataset->centerLED << "," << dataset->cropX << "," << dataset->cropY << "," << dataset->Np << "," << dataset->du << "," << dataset->bk1cropX << "," << dataset->bk1cropY << "," << dataset->bk2cropX << "," << dataset->bk2cropY << "," << dataset->bgThreshold << ";" << std::endl;
   
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      fileObj2 << dataset->NALedPatternStackX.at(imgIdx) << "," ;
   }
   fileObj2  <<";"<<std::endl;
   
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      fileObj2 << dataset->NALedPatternStackY.at(imgIdx) << "," ;
   }
   fileObj2  <<";"<<std::endl;
   
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      fileObj2 << dataset->illuminationNAList.at(imgIdx) << "," ;
   }
   fileObj2  <<";"<<std::endl;
   
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      fileObj2 << dataset->sortedNALedPatternStackX.at(imgIdx) << "," ;
   }
   fileObj2  <<";"<<std::endl;
   
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      fileObj2 << dataset->sortedNALedPatternStackY.at(imgIdx) << "," ;
   }
   fileObj2  <<";"<<std::endl;
   
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      fileObj2 << dataset->sortedIndicies.at(imgIdx) << "," ;
   }
   fileObj2 <<";" << std::endl;
   
   fileObj2.close();


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



void darkCurrentSubtraction(FPM_Dataset dataset)
{
   
}

int main(int argc, char** argv )
{
   // Parameters from the .m file
   cout << "working!" <<endl;
   uint16_t Np = 90;
   FPM_Dataset mDataset;
   mDataset.datasetRoot = "/home/zfphil/Dropbox/FP_mono_nofilter/";
   mDataset.ledCount = 508;
   mDataset.pixelSize = 6.5;
   mDataset.objectiveMag = 4*2;
   mDataset.objectiveNA = 0.2;
   mDataset.color = false;
   mDataset.centerLED = 249;
   mDataset.lambda = 0.5;
   mDataset.ps_eff = mDataset.pixelSize / (float) mDataset.objectiveMag;
   mDataset.du= (1/mDataset.ps_eff)/(float) Np;
   cout << mDataset.du << endl;
  
   
   char fileName[FILENAME_LENGTH];
   sprintf(fileName,"%s%04d.tif",filePrefix.c_str(),mDataset.centerLED);
   cout << mDataset.datasetRoot + fileName <<endl;

   Mat centerImg = imread(mDataset.datasetRoot + fileName, -1);
   
   int16_t cropX = 1248;
   int16_t cropY = 1020;
   //cv::Mat croppedRef(centerImg, cv::Rect(cropX,cropY,Np,Np));
  // showImg(croppedRef);
   
   mDataset.Np = Np;
   mDataset.cropX = cropX;
   mDataset.cropY = cropY;
   
   mDataset.bk1cropX = 1170;
   mDataset.bk1cropY = 1380; 
   mDataset.bk2cropX = 1080; 
   mDataset.bk2cropY = 700; 
   
   mDataset.bgThreshold = 1000;
   
   loadDataset(&mDataset);
   
   //saveFullDataset(&mDataset, "tmpDataset/");
   /*
   showImg(mDataset.imageStack.at(249).Image);
   imwrite("tmp.tiff",mDataset.imageStack.at(249).Image);
  // fpmBackgroundSubtraction(&mDataset);
  */
   
}
