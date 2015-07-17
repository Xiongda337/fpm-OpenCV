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
#include <vector>

//#include "include/rapidjson"
#include "domeHoleCoordinates.h"

using namespace std;
using namespace cv;

#define FILENAME_LENGTH 128
#define FILE_HOLENUM_DIGITS 4

string filePrefix = "iLED_";

using CP = std::complex<float>;

// Debug flags
bool preprocessDebug = false;
bool runDebug = false;

class FPMimg{
  public:
        cv::Mat Image;
        cv::Mat Objfcrop;
        cv::Mat ObjfcropP;
        cv::Mat ObjcropP;
        cv::Mat Objfup;
        int led_num;
        float sinTheta_x;
        float sinTheta_y;
        float vled;
        float uled;
        int16_t idx_u;
        int16_t idx_v;
        float illumination_na;
        int16_t bg_val;
        int16_t pupilShiftX;
        int16_t pupilShiftY;
        int16_t cropYStart;
        int16_t cropYEnd;
        int16_t cropXStart;
        int16_t cropXEnd;
        
};

class FPM_Dataset{
  public:
        string                datasetRoot;
        std::vector<FPMimg>   imageStack;
        uint16_t              ledCount;
        float                 pixelSize;           // pixel size in microns
        float                 objectiveMag;
        float                 objectiveNA;
        float                 maxIlluminationNA;
        float                 lambda;              // wavelength in microns
        bool                  color;                // flag for data acquired on color camera
        int16_t               centerLED = 249;     // Closest LED to center of Array
        int16_t               cropX;
        int16_t               cropY;
        int16_t               Np;                  // ROI Size
        int16_t               Np_padded;           // Zero-padded ROI
        int16_t               Mcrop;
        int16_t               Ncrop;
        int16_t               Nimg;
        int16_t               Nlarge;
        int16_t               Mlarge;
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
        float                 ps; // Recovered Pixel Size
        
        // FPM Specific
        float                 delta1;
        float                 delta2;
        cv::Mat               obj;                      // Reconstructed object, real space, full res
        cv::Mat               objCrop;                  // Reconstructed object, real space, cropped
        cv::Mat               objF;                     // Reconstructed object, Fourier space, full res
        cv::Mat               objFCrop;                 // Reconstructed object, Fourier space, cropped
        cv::Mat               pupil;                     // Reconstructed pupil, Fourier Space
        cv::Mat               pupilSupport;             // Binary mask for pupil support, Fourier space
        int16_t               itrCount = 10;            // Iteration Count
        
        float                 eps = 0.0000000001;
};

/*/Check if file exists
inline bool exists_test (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}*/



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

void circularShift(Mat img, Mat result, int x, int y){
    int w = img.cols;
    int h  = img.rows;

    int shiftR = x % w;
    int shiftD = y % h;
    
    if (shiftR < 0)//if want to shift in -x direction
        shiftR += w;
    
    if (shiftD < 0)//if want to shift in -y direction
        shiftD += h;

    cv::Rect gate1(0, 0, w-shiftR, h-shiftD);//rect(x, y, width, height)
    cv::Rect out1(shiftR, shiftD, w-shiftR, h-shiftD);
    
    cv::Rect gate2(w-shiftR, 0, shiftR, h-shiftD); 
    cv::Rect out2(0, shiftD, shiftR, h-shiftD);
    
    cv::Rect gate3(0, h-shiftD, w-shiftR, shiftD);
    cv::Rect out3(shiftR, 0, w-shiftR, shiftD);
    
    cv::Rect gate4(w-shiftR, h-shiftD, shiftR, shiftD);
    cv::Rect out4(0, 0, shiftR, shiftD);
   
    cv::Mat shift1 = img ( gate1 );
    cv::Mat shift2 = img ( gate2 );
    cv::Mat shift3 = img ( gate3 );
    cv::Mat shift4 = img ( gate4 );

    shift1.copyTo(cv::Mat(result, out1));//copyTo will fail if any rect dimension is 0
    shift2.copyTo(cv::Mat(result, out2));
    shift3.copyTo(cv::Mat(result, out3));
    shift4.copyTo(cv::Mat(result, out4));
}

void maxComplexReal(cv::Mat& m, string label)
{
      Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
      split(m,planes);
      double minVal, maxVal;
      cv::minMaxLoc(planes[0], &minVal, &maxVal);
      cout << "Max/Min values of " <<label << " are: " << maxVal << ", " << minVal << endl;
      
}

int loadDataset(FPM_Dataset *dataset) {
	DIR *dir;
	struct dirent *ent;
	Mat fullImg;
	Mat fullImgComplex;
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
    		      //Mat planes[] = {Mat_<float>(fullImg(cv::Rect(dataset->cropX,dataset->cropY,dataset->Np,dataset->Np)).clone()), Mat::zeros(dataset->Np,dataset->Np, CV_32F)};
    		      // merge(planes, 2, fullImgComplex);      
    		      //  currentImage.Image = fullImgComplex;
    		      
    		      currentImage.Image = fullImg(cv::Rect(dataset->cropX,dataset->cropY,dataset->Np,dataset->Np)).clone();
          		currentImage.sinTheta_x = sin(atan2(domeHoleCoordinates[currentImage.led_num-1][0], domeHoleCoordinates[currentImage.led_num-1][2]));
          		currentImage.sinTheta_y = sin(atan2(domeHoleCoordinates[currentImage.led_num-1][1] , domeHoleCoordinates[currentImage.led_num-1][2]));
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
          		
          		currentImage.pupilShiftX = (int16_t) round( currentImage.sinTheta_x / dataset->lambda * dataset->ps * dataset->Nlarge) ; // Deal with MATLAB indexing
          		currentImage.pupilShiftY = (int16_t) round( currentImage.sinTheta_y / dataset->lambda * dataset->ps * dataset->Mlarge); // Deal with MATLAB indexing
          		
          		currentImage.cropXStart = (int16_t)round(dataset->Nlarge/2) + currentImage.pupilShiftX - (int16_t)round(dataset->Ncrop/2);
          		currentImage.cropXEnd = (int16_t)round(dataset->Nlarge/2) + currentImage.pupilShiftX + (int16_t)round(dataset->Ncrop/2) - 1;
          		
          		currentImage.cropYStart = (int16_t)round(dataset->Mlarge/2) + currentImage.pupilShiftY - (int16_t)round(dataset->Ncrop/2);
          		currentImage.cropYEnd = (int16_t)round(dataset->Mlarge/2) + currentImage.pupilShiftY + (int16_t)round(dataset->Ncrop/2) - 1;
          		
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
             		std::cout << "   sintheta_x is: " << currentImage.sinTheta_x << ", sintheta_y is: " << currentImage.sinTheta_y << std::endl;
             		std::cout << "   Image Size is: " << currentImage.Image.rows << " x " << currentImage.Image.cols << std::endl;
             		std::cout << "   cartx is: " << domeHoleCoordinates[currentImage.led_num-1][0] << std::endl;
             		std::cout << "   carty is: " << domeHoleCoordinates[currentImage.led_num-1][1] << std::endl;
             		std::cout << "   cartz is: " << domeHoleCoordinates[currentImage.led_num-1][2] << std::endl;
             		std::cout << "   atan(cartx) : " << atan(domeHoleCoordinates[currentImage.led_num-1][0] / domeHoleCoordinates[currentImage.led_num-1][2]) << std::endl;
             		std::cout << "   atan(carty) : " << atan(domeHoleCoordinates[currentImage.led_num-1][1] / domeHoleCoordinates[currentImage.led_num-1][2]) << std::endl;
             		
             	   std::cout << "   sin(atan(cartx)) : " << sin(atan(domeHoleCoordinates[currentImage.led_num-1][0] / domeHoleCoordinates[currentImage.led_num-1][2])) << std::endl;
             		std::cout << "   sin(atan(carty)) : " << sin(atan(domeHoleCoordinates[currentImage.led_num-1][1] / domeHoleCoordinates[currentImage.led_num-1][2])) << std::endl;
             		
             		std::cout << "   uled : " << currentImage.uled << std::endl;
             		std::cout << "   vled : " << currentImage.vled << std::endl;
             		
             		std::cout << "   idx_u : " << currentImage.idx_u << std::endl;
             		std::cout << "   idx_v : " << currentImage.idx_v << std::endl;
             		
             		std::cout << "   pupilShiftX : " << currentImage.pupilShiftX << std::endl;
             		std::cout << "   pupilShiftY : " << currentImage.pupilShiftY << std::endl;
             		
             		std::cout << "   cropXStart : " << currentImage.cropXStart << std::endl;
             		std::cout << "   cropXEnd : " << currentImage.cropXEnd << std::endl;
             		
             		std::cout << "   cropYStart : " << currentImage.cropYStart << std::endl;
             		std::cout << "   cropYEnd : " << currentImage.cropYEnd << std::endl;
             		
 
                  std::cout << "   illumination na: " << currentImage.illumination_na <<std::endl;
                  std::cout << std::endl<< std::endl;
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

void complexConj(const cv::Mat& m, cv::Mat& output)
{
    Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
    cv::split(m, planes);
    planes[1] = -1.0 * planes[1];
    cv::merge(planes, 2, output);
}

// Returns the absolute value of a complex-valued matrix (channel 1 = abs, channel 2 = zeros)
void complexAbs(const cv::Mat& m, cv::Mat& output)
{
    Mat planes[] = {Mat::zeros(m.rows, m.cols, CV_64F),Mat::zeros(m.rows, m.cols, CV_64F)};
    cv::pow(m,2,output);
    cv::split(output,planes);
    cv::sqrt(planes[0] + planes[1], planes[0]);
    planes[1] = Mat::zeros(m.rows, m.cols, CV_64F);
    cv::merge(planes,2,output);
}

void complexMultiply(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
{
   Mat outputPlanes[] = {Mat::zeros(m1.rows, m1.cols, m1.type()),Mat::zeros(m1.rows, m1.cols, CV_64F)};
   Mat tmpMat(m1.rows,m1.cols,CV_64F);
   std::vector<cv::Mat> comp1;
   std::vector<cv::Mat> comp2;
   cv::split(m1,comp1);
   cv::split(m2,comp2);
   
   // (a+bi) * (c+di) = ac - bd + (ad+bc) * i
   // Real Part
   cv::multiply(comp1[0], comp2[0], tmpMat);
   outputPlanes[0] = tmpMat.clone();

   cv::multiply(comp1[1], comp2[1], tmpMat);
   outputPlanes[0] = outputPlanes[0] - tmpMat;
   
   // Imaginary Part
   cv::multiply(comp1[0], comp2[1], tmpMat);
   outputPlanes[1] = tmpMat.clone();
   cv::multiply(comp1[1], comp2[0], tmpMat);
   outputPlanes[1] = outputPlanes[1] + tmpMat;
   
   merge(outputPlanes,2,output);
}

void complexScalarMultiply(double scaler, cv::Mat& m, cv::Mat output)
{
   Mat inputPlanes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, CV_64F)};
   Mat outputPlanes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, CV_64F)};
   cv::split(m,inputPlanes);
   outputPlanes[0] = scaler * inputPlanes[0];
   outputPlanes[1] = scaler * inputPlanes[1];
   merge(outputPlanes,2,output);
}

void complexDivide(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
{
   Mat outputPlanes[] = {Mat::zeros(m1.rows, m1.cols, m1.type()),Mat::zeros(m1.rows, m1.cols, CV_64F)};
   Mat numerator(m1.rows,m1.cols,CV_64F);
   Mat denominator(m1.rows,m1.cols,CV_64F);
   Mat tmpMat(m1.rows,m1.cols,CV_64F); Mat tmpMat2(m1.rows,m1.cols,CV_64F);
   
   std::vector<cv::Mat> comp1;
   std::vector<cv::Mat> comp2;
   cv::split(m1,comp1);
   cv::split(m2,comp2);
   
   // (a+bi) / (c+di) = (ac+bd) / (c^2+d^2) + (bc-ad) / (c^2+d^2) * i
   
   // Real Part
   cv::multiply(comp1[0], comp2[0],tmpMat);       // ac
   cv::multiply(comp1[1], comp2[1], tmpMat2);     // bd
   numerator = tmpMat + tmpMat2;                  // ac+bd
   
   cv::multiply(comp2[0], comp2[0],tmpMat);       // c^2
   cv::multiply(comp2[1], comp2[1], tmpMat2);     // d^2
   denominator = tmpMat + tmpMat2;                  // c^2+d^2
   
   cv::divide(numerator, denominator , outputPlanes[0]); // (ac+bd) / (c^2+d^2) 
   
   // Imaginary Part
   cv::multiply(comp1[1], comp2[0],tmpMat);       // bc
   cv::multiply(comp1[0], comp2[1], tmpMat2);     // ad
   numerator = tmpMat - tmpMat2;                  // bc-ad
   
   // Same denominator
   cv::divide(numerator, denominator , outputPlanes[1]); // (ac+bd) / (c^2+d^2) 
   
   merge(outputPlanes,2,output);
}

// Perform matrix inverse
void complexInverse(const cv::Mat& m, cv::Mat& inverse)
{
   Mat one1 = Mat::ones(m.rows,m.cols,m.type());
   complexDivide(one1, m ,inverse);
}

cv::Mat fftShift(cv::Mat m)
{
      cv::Mat shifted = cv::Mat(m.cols,m.rows,m.type());
      circularShift(m, shifted, std::ceil((double) m.cols/2), std::ceil((double) m.rows/2));
      return shifted;
}

cv::Mat ifftShift(cv::Mat m)
{
      cv::Mat shifted = cv::Mat(m.cols,m.rows,m.type());
      circularShift(m, shifted, std::floor((double) m.cols/2), std::floor((double)m.rows/2));
      return shifted;
}

// Opencv fft implimentation
void fft2(cv::Mat& input, cv::Mat& output)
{
   cv::dft(input, output, DFT_COMPLEX_OUTPUT);
}
 
// Opencv ifft implimentation
void ifft2(cv::Mat& input, cv::Mat& output)
{
   cv::dft(input, output, DFT_INVERSE | DFT_COMPLEX_OUTPUT | DFT_SCALE); // Real-space of object
}

void complex_imwrite(string fname, Mat& m1)
{
   Mat outputPlanes[] = {Mat::zeros(m1.rows, m1.cols, m1.type()), Mat::zeros(m1.rows, m1.cols, m1.type()),Mat::zeros(m1.rows, m1.cols, m1.type())};
   Mat inputPlanes[] = {Mat::zeros(m1.rows, m1.cols, m1.type()),Mat::zeros(m1.rows, m1.cols, m1.type())};
   
   cv::split(m1,inputPlanes);
   outputPlanes[0] = inputPlanes[0];
   outputPlanes[1] = inputPlanes[1];
   cv::Mat outMat;
   cv::merge(outputPlanes,3,outMat);
   imwrite(fname,outMat);
}

void onMouse( int event, int x, int y, int, void* param )
{

    Mat* imgPtr = (Mat*) param;
    Mat image;
    imgPtr->copyTo(image);
    
    //Mat image = imgPtr.clone();
    
    switch (event)
    {
       case CV_EVENT_LBUTTONDOWN:
       {
          Mat planes[] = {Mat::zeros(image.rows, image.cols, image.type()),Mat::zeros(image.rows, image.cols, image.type())};
          split(image, planes);    
        
          printf("x:%d y:%d: %f + %fi\n", 
          x, y, 
          planes[0].at<double>(y,x), 
          planes[1].at<double>(y,x));
          break;
       }
       case CV_EVENT_RBUTTONDOWN:
       {
          Mat planes[] = {Mat::zeros(image.rows, image.cols, image.type()),Mat::zeros(image.rows, image.cols, image.type())};
          split(image, planes);      
          double minVal, maxVal;
          cv::minMaxLoc(planes[0], &minVal, &maxVal);
          std::cout << "Max/Min values of real part are: " << maxVal << ", " << minVal << std::endl;
          cv::minMaxLoc(planes[1], &minVal, &maxVal);
          std::cout << "Max/Min values of imaginary part are: " << maxVal << ", " << minVal << std::endl << std::endl;
       break;
       }
       default:
         return;
   }
}

void showImgMag(Mat m, string windowTitle)
{

   Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
   cv::Mat m2 = fftShift(m);
   split(m2, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
   magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
   Mat magI = planes[0];

   magI += Scalar::all(1);                    // switch to logarithmic scale
   log(magI, magI);

   // crop the spectrum, if it has an odd number of rows or columns
   //magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
   normalize(magI, magI, 0, 1, CV_MINMAX);
   namedWindow(windowTitle, WINDOW_NORMAL);
   setMouseCallback(windowTitle, onMouse, &m2);
   imshow(windowTitle, magI);
   waitKey();
   cv::destroyAllWindows();
}

void showImgFourier(Mat m, string windowTitle)
{
   Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
   cv::Mat m2 = fftShift(m);
   split(m2, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
   magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
   Mat magI = planes[0];

   magI += Scalar::all(1);                    // switch to logarithmic scale
   log(magI, magI);

   // crop the spectrum, if it has an odd number of rows or columns
   //magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
   normalize(magI, magI, 0, 1, CV_MINMAX);
   namedWindow(windowTitle, WINDOW_NORMAL);
   setMouseCallback(windowTitle, onMouse, &m2);
   imshow(windowTitle, magI);
   waitKey();
   cv::destroyAllWindows();
}

void showImgObject(Mat m, string windowTitle)
{
   Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
   split(m, planes);
   normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
   normalize(planes[1], planes[1], 0, 1, CV_MINMAX);
   namedWindow(windowTitle + " REAL", WINDOW_NORMAL);
   setMouseCallback(windowTitle + " REAL", onMouse, &planes[0]);
   imshow(windowTitle + " REAL", planes[0]);
   namedWindow(windowTitle + " IMAG", WINDOW_NORMAL);
   setMouseCallback(windowTitle + " IMAG", onMouse, &planes[1]);
   imshow(windowTitle + " IMAG", planes[1]);
   waitKey();
   cv::destroyAllWindows();
}

void showImg(Mat m, string windowTitle)
{
   Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
   split(m, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
   magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
   Mat magI = planes[0];

   magI += Scalar::all(1);                    // switch to logarithmic scale
   log(magI, magI);

   // crop the spectrum, if it has an odd number of rows or columns
   //magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
   normalize(magI, magI, 0, 1, CV_MINMAX);
   namedWindow(windowTitle, WINDOW_NORMAL);
   setMouseCallback(windowTitle, onMouse, &m);
   imshow(windowTitle, magI);
   waitKey();
   cv::destroyAllWindows();
}

void printComplexPixelValue(cv::Mat& m, string label, int16_t pxNum)
{
      Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
      split(m,planes);
      cout << "Pixel " << pxNum <<"x"<<pxNum<<" of " <<label << " is: " << planes[0].at<double>(pxNum,pxNum) << " + " << planes[1].at<double>(pxNum,pxNum) << "i " << endl; 
}

void run(FPM_Dataset * dataset)
{
   // Make dummy pointers to save space
   Mat * objF = &dataset->objF;
   
   // Initilize Matricies
   Mat tmpMat1;
   Mat tmpMat2;
   Mat tmpMat3;
   Mat pupilAbs;
   Mat pupilConj;
   double q, pupilMax;
   double p, objf_max;
   Mat objfcrop_abs;
   Mat objfcrop_conj;
   
   Mat Objfcrop_abs; Mat Objfcrop_abs_sq; Mat Objf_abs; Mat Objfcrop_conj; Mat Objfcrop_abs_conj;

   // Initialize pupil function
   Mat planes[] = {Mat::zeros(dataset->Np,dataset->Np, CV_64F), Mat::zeros(dataset->Np,dataset->Np, CV_64F)};
   planes[0] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
   planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);

   cv::Point center(cvRound(dataset->Np/2),cvRound(dataset->Np/2));
   int16_t naRadius = (int16_t) ceil(dataset->objectiveNA * dataset->ps_eff * dataset->Np / dataset->lambda);
   //circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
   cv::circle(planes[0], center, naRadius ,cv::Scalar(1.0), -1, 8, 0);
   
   // FFTshift the pupil so it is consistant with object FT   
   planes[0] = fftShift(planes[0]);
   
   merge(planes, 2, dataset->pupil);
   merge(planes, 2, dataset->pupilSupport);
   
   // Initialize FT of reconstructed object with center led image
   Mat complexI;
   planes[0] = Mat_<double>(dataset->imageStack.at(dataset->sortedIndicies.at(1)).Image);
   cv::sqrt(planes[0], planes[0]); // Convert to amplitude
   planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
   
   merge(planes, 2, complexI);
   //dft(complexI, complexI, DFT_SCALE | DFT_COMPLEX_OUTPUT);
   fft2(complexI,complexI);
   
   complexI = fftShift(complexI); // Shift to center
   //complexMultiply(complexI,fftShift(dataset->pupilSupport),complexI);
   
   showImgFourier(complexI,"Initialized FFT (Should be fftshifted)");
   
   planes[0] = Mat::zeros(dataset->Nlarge,dataset->Mlarge, CV_64F);
   planes[1] = Mat::zeros(dataset->Nlarge,dataset->Mlarge, CV_64F);
   merge(planes,2,dataset->objF);
   
   complexI.copyTo(cv::Mat(dataset->objF, cv::Rect((int16_t)round(dataset->Mlarge/2) - (int16_t)round(dataset->Ncrop/2),(int16_t)round(dataset->Mlarge/2) - (int16_t)round(dataset->Ncrop/2),dataset->Np,dataset->Np)));
    
   // Shift to un-fftshifted position
   dataset->objF = fftShift(dataset->objF);
   
   showImgFourier(dataset->objF,"Initialized ObjF");
   
   
   for (int16_t itr = 1; itr <= dataset->itrCount; itr++)
   {
      for (int16_t imgIdx = 1; imgIdx <= dataset->ledCount; imgIdx++) // 
      {
      int16_t ledNum = dataset->sortedIndicies.at(imgIdx);
      //cout<<ledNum<<endl;
      
      FPMimg * currImg;
      currImg = & dataset->imageStack.at(ledNum);
     
      // Update Fourier space, multply by pupil (P * O)
      tmpMat2 = fftShift(dataset->objF); // Shifted Object spectrum (at center)
      currImg->Objfcrop = fftShift(tmpMat2(cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np))); // Take ROI from shifted object spectrum
      complexMultiply(currImg->Objfcrop, dataset->pupil, currImg->ObjfcropP);
      //tmpMat2 = ifftShift(currImg->ObjfcropP);
      ifft2(currImg->ObjfcropP,currImg->ObjcropP);
      if (runDebug)
      {
         std::cout << "NEW LED" <<std::endl;
         showImgFourier(currImg->Objfcrop,"currImg->Objfcrop");
         showImgFourier(currImg->ObjfcropP,"currImg->ObjfcropP");
         showImgObject(currImg->ObjcropP,"currImg->ObjcropP");
      }
       
      // Replace Amplitude
      currImg->Image.convertTo(tmpMat1,CV_64FC1);
      planes[0] = tmpMat1;
      planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
      cv::merge(planes,2,tmpMat1);
      cv::sqrt(tmpMat1,tmpMat2); // Works because tmpMat is real-valued (complex portion is zero)
      
      complexAbs(currImg->ObjcropP + dataset->eps, tmpMat3);
      complexDivide(currImg->ObjcropP, tmpMat3, tmpMat1);
      complexMultiply(tmpMat1, tmpMat2 ,tmpMat3);
      fft2(tmpMat3,currImg->Objfup);
      
      if(runDebug)
      {
           showImgObject(tmpMat2,"Amplitude of Input Image");
           showImgObject(tmpMat3,"Image with amplitude   replaced");
           showImgFourier(currImg->Objfup,"currImg->Objfup");
      }
      
      ///////// Alternating Projection Method - Object ///////////
      // MATLAB: Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) = Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) + abs(Pupil).*conj(Pupil).*(Objfup-ObjfcropP)/max(abs(Pupil(:)))./(abs(Pupil).^2+delta2);
      
      // Numerator 
      Mat pupil_abs; Mat pupil_abs_sq; Mat pupil_conj; Mat numerator; Mat denomSum;
      complexAbs(dataset->pupil,pupil_abs);
      complexConj(dataset->pupil, pupil_conj);
      complexMultiply(pupil_abs, pupil_conj, tmpMat1);
      complexMultiply(currImg->Objfup - currImg->ObjfcropP, tmpMat1, numerator);
      
      // Denominator
      double p; double pupil_abs_max;
      cv::minMaxLoc(pupil_abs, &p, &pupil_abs_max);
      complexMultiply(pupil_abs,pupil_abs,pupil_abs_sq);
      denomSum = pupil_abs_sq + dataset->delta2;
      complexDivide(numerator, denomSum * pupil_abs_max, tmpMat2);
      
      if(runDebug)
      {
           showImgFourier(numerator,"Object update Numerator");
           showImgFourier(tmpMat2,"Object update Denominator");
      }
      
      Mat objF_centered;
      objF_centered = fftShift(dataset->objF);
      
      Mat objF_cropped = cv::Mat(objF_centered, cv::Rect(currImg->cropXStart, currImg->cropYStart, dataset->Np, dataset->Np));
      tmpMat1 = fftShift(tmpMat2) + objF_cropped;
      
      if(runDebug)
      {
           showImgFourier(objF_cropped,"Origional Object Spectrum to be updated");
           showImgFourier(fftShift(tmpMat2),"Object spectrum update incriment");
      }

      // Replace the region in objF
      tmpMat1.copyTo(cv::Mat(objF_centered, cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np)));
      dataset->objF = fftShift(objF_centered);
      
      if(runDebug)
      {
           showImgFourier(fftShift(tmpMat1),"Cropped updated object spectrum");
           showImgFourier(dataset->objF,"Full updated object spectrum");
      }
      
      ////// PUPIL UPDATE ///////
      // Numerator 
      //showImgFourier(currImg->Objfcrop,"currImg->Objfcrop");
      complexAbs(currImg->Objfcrop, Objfcrop_abs);
      complexAbs(dataset->objF, Objf_abs);
      //showImgFourier(dataset->objF,"dataset->objF");
      
      complexConj(currImg->Objfcrop, Objfcrop_conj); 
      complexMultiply(Objfcrop_abs, Objfcrop_conj, tmpMat1);
      complexMultiply(currImg->Objfup - currImg->ObjfcropP, tmpMat1, numerator);
      
      // Denominator
      double Objf_abs_max;
      cv::minMaxLoc(Objf_abs, &p, &Objf_abs_max);
      complexMultiply(Objfcrop_abs,Objfcrop_abs,Objfcrop_abs_sq);
      denomSum = Objfcrop_abs_sq + dataset->delta1;
      complexDivide(numerator, denomSum * Objf_abs_max, tmpMat2);
      complexMultiply(tmpMat2,dataset->pupilSupport, tmpMat2);
      
      dataset->pupil += tmpMat2; 
      }
      cout<<"Iteration " << itr << " Completed!" <<endl;
      dft(dataset->objF ,dataset->objCrop, DFT_INVERSE | DFT_SCALE);
    
   }
     showImgObject((dataset->objCrop), "Object");
     showImgFourier((dataset->objF),"Object Spectrum");
     showImgFourier((dataset->pupil),"Pupil");
      //showImgMag(fftShift(dataset->pupil),"Pupil"); 
}

int main(int argc, char** argv )
{
   // Parameters from the .m file
   uint16_t Np = 90;
   FPM_Dataset mDataset;
   mDataset.datasetRoot = "/home/zfphil/Dropbox/FP_mono_nofilter/";
   mDataset.ledCount = 508;
   mDataset.pixelSize = 6.5;
   mDataset.objectiveMag = 4*2;
   mDataset.objectiveNA = 0.2;
   mDataset.maxIlluminationNA = 0.7604;
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
   
   int16_t resImprovementFactor = (int16_t) ceil(2*mDataset.ps_eff*(mDataset.maxIlluminationNA+mDataset.objectiveNA)/mDataset.lambda); // Ask Li-Hao what this does exactly, and what to call it.
   mDataset.bgThreshold = 1000;
   mDataset.Mcrop = mDataset.Np;
   mDataset.Ncrop = mDataset.Np;
   mDataset.Nimg  = mDataset.ledCount;
   mDataset.Nlarge = mDataset.Ncrop * resImprovementFactor;
   mDataset.Mlarge = mDataset.Mcrop * resImprovementFactor;
   mDataset.ps = mDataset.ps_eff / (float)resImprovementFactor;
   mDataset.delta1 = 5;
   mDataset.delta2 = 100;
   
   /* TESTING COMPLEX MAT FUNCTIONS 
   Mat testMat1;
   Mat testMat2;
   Mat testMat3;
   Mat testMat4;
   
   Mat conj1; Mat abs1; Mat tmp1; Mat numeratorMat; Mat abssq; Mat abs2;
   
   Mat test1[] = {Mat::ones(3, 3, CV_64F)*-1, Mat::ones(3, 3, CV_64F)*2};
   Mat test2[] = {Mat::ones(3, 3, CV_64F)*3,Mat::ones(3, 3, CV_64F)*-4};
   Mat test3[] = {Mat::ones(3, 3, CV_64F)*5,Mat::ones(3, 3, CV_64F)*-6};
   Mat test4[] = {Mat::ones(3, 3, CV_64F)*7,Mat::ones(3, 3, CV_64F)*-8};
   
   Mat out[] = {Mat::ones(3, 3, CV_64F)*7,Mat::ones(3, 3, CV_64F)*-8};
   
   cv::merge(test1,2,testMat1);
   cv::merge(test2,2,testMat2);
   cv::merge(test3,2,testMat3);
   cv::merge(test4,2,testMat4);
   Mat outputMat; Mat tmp;

   complexAbs(testMat1,abs1);
   complexAbs(testMat4,abs2);
   complexConj(testMat1,conj1);
   complexMultiply(abs1,conj1,tmp1);
   
   // good here
   
   complexMultiply(testMat2-testMat3,tmp1,numeratorMat);
   
   double p; double max; double delta1 = 1000;
   cv::minMaxLoc(abs2, &p, &max);
   cout << "abs2 Max: " << max << endl;
   
   complexMultiply(abs1,abs1,abssq);


   
   cv::split(numeratorMat,test1);
   cout << "numerator (real) = " << endl << " "  << test1[0]<< ",  (imag) = " << endl << " "  << test1[1] << endl << endl;
   
   // NUMERATOR IS GOOD
   
   Mat sum2 = (abssq+delta1);
   
   cv::split(sum2,test1);
   cout << "abssq+delta1 = " << endl << " "  << test1[0]<< ",  (imag) = " << endl << " "  << test1[1] << endl << endl;
   
   
   // NEED TO ADD AND MULTIPLY SEPERATLY
   cv::split(sum2 * max,test1);
   cout << "denominator inverse(NORMAL) (real) = " << endl << " "  << test1[0]<< ",  (imag) = " << endl << " "  << test1[1] << endl << endl;
   
   Mat inverse;
   Mat one1 = Mat::ones(testMat1.rows,testMat1.cols,testMat1.type());
   complexDivide(one1,sum2 * max,inverse);

   cv::split(inverse,out);
   cout << "Inverse of Denom (real) = " << endl << " "  << out[0] <<",   (imag) = " << endl << " "  << out[1] << endl << endl;
   
      complexDivide(numeratorMat, sum2 * max, outputMat);
   
   
   cv::split(testMat1,test1);
   cv::split(testMat2,test2);
   cv::split(testMat3,test3);
   cv::split(testMat4,test4);
   
   cout << "objfcrop 1 (real) = " << endl << " "  << test1[0]<< ",  (imag) = " << endl << " "  << test1[1] << endl << endl;
   cout << "objfup 2 (real) = " << endl << " "  << test2[0] << ",   (imag) = " << endl << " "  << test2[1] << endl << endl;
   cout << "objfcropp 3 (real) = " << endl << " "  << test3[0] << ",   (imag) = " << endl << " "  << test3[1] << endl << endl;
   cout << "objf 4 (real) = " << endl << " "  << test4[0] << ",   (imag) = " << endl << " "  << test4[1] << endl << endl;
   
   cv::split(outputMat,out);
   cout << "Output (real) = " << endl << " "  << out[0] <<",   (imag) = " << endl << " "  << out[1] << endl << endl;
   
   */
   
   loadDataset(&mDataset);
   
   run(&mDataset);
   
   //saveFullDataset(&mDataset, "tmpDataset/");
   /*
   showImg(mDataset.imageStack.at(249).Image);
   imwrite("tmp.tiff",mDataset.imageStack.at(249).Image);
  // fpmBackgroundSubtraction(&mDataset);
  */
   
}
