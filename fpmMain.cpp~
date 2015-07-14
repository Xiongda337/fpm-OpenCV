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

bool preprocessDebug = false;
bool runDebug = false;
int16_t debugPxVal = 40;

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
        int16_t               itrCount = 20;            // Iteration Count
        
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
          		
          		currentImage.pupilShiftX = (int16_t) round( currentImage.sinTheta_x / dataset->lambda * dataset->ps * dataset->Nlarge);
          		currentImage.pupilShiftY = (int16_t) round( currentImage.sinTheta_y / dataset->lambda * dataset->ps * dataset->Mlarge);
          		
          		currentImage.cropXStart = (int16_t)round(dataset->Nlarge/2) + 1 + currentImage.pupilShiftX - (int16_t)round(dataset->Ncrop/2);
          		currentImage.cropXEnd = (int16_t)round(dataset->Nlarge/2) + 1 + currentImage.pupilShiftX + (int16_t)round(dataset->Ncrop/2) - 1;
          		
          		currentImage.cropYStart = (int16_t)round(dataset->Mlarge/2) + 1 + currentImage.pupilShiftY - (int16_t)round(dataset->Ncrop/2);
          		currentImage.cropYEnd = (int16_t)round(dataset->Mlarge/2) + 1 + currentImage.pupilShiftY + (int16_t)round(dataset->Ncrop/2) - 1;
          		
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

void complexInverse(const cv::Mat& m, cv::Mat& inverse)
{
   cv::divide(1.0,m,inverse);
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
    Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, CV_64F)};
    cv::pow(m,2,m);
    cv::split(m,planes);
    cv::sqrt(planes[0]+planes[1], planes[0]);
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

void complexDivide(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
{
   Mat inverse;
   complexInverse(m2,inverse);
   complexMultiply(m1, inverse, output);
}

cv::Mat fftShift(cv::Mat m)
{
      cv::Mat shifted = cv::Mat(m.cols,m.rows,m.type());
      circularShift(m, shifted, round(m.cols/2), round(m.rows/2));
      return shifted;
}

cv::Mat ifftShift(cv::Mat m)
{
      cv::Mat shifted = cv::Mat(m.cols,m.rows,m.type());
      circularShift(m, shifted, round(m.cols/-2), round(m.rows/-2));
      return shifted;
}

void showImg(Mat img, string windowTitle)
{
   Mat imgN;
   normalize(img, imgN, 0, 1, CV_MINMAX);
   namedWindow(windowTitle, WINDOW_NORMAL);
   //setMouseCallback( windowTitle, onMouse, &img );
   imshow(windowTitle, imgN);
   waitKey();
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
        
          printf("%d %d: %f + %fi\n", 
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
   cv::destroyAllWindows();
   Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
   cv::Mat m2 = m;// fftShift(m);
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
   
   // Initialize FT of reconstructed object with center led image
   Mat complexI;
   Mat planes[] = {Mat_<double>(dataset->imageStack.at(dataset->sortedIndicies.at(1)).Image), Mat::zeros(dataset->Np,dataset->Np, CV_64F)};
   merge(planes, 2, complexI);
   tmpMat1 = fftShift(complexI);
   cv::sqrt(tmpMat1,tmpMat1); //Intensity to amplitude
   dft(tmpMat1,tmpMat1,DFT_COMPLEX_OUTPUT);
   
   planes[0] = Mat::zeros(dataset->Nlarge,dataset->Mlarge, CV_64F);
   planes[1] = Mat::zeros(dataset->Nlarge,dataset->Mlarge, CV_64F);
   merge(planes,2,dataset->objF);
   tmpMat2 = fftShift(tmpMat1);
   
   tmpMat2.copyTo(cv::Mat(dataset->objF, cv::Rect(dataset->imageStack.at(dataset->sortedIndicies.at(1)).cropXStart,dataset->imageStack.at(dataset->sortedIndicies.at(1)).cropYStart,dataset->Np,dataset->Np)));
   
   tmpMat1 = fftShift(dataset->objF);
   dataset->objF = tmpMat1.clone();
   
   // Initialize pupil function
   planes[0] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
   planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);

   cv::Point center(cvRound(dataset->Np/2),cvRound(dataset->Np/2));
   int16_t naRadius = (int16_t) ceil(dataset->objectiveNA * dataset->ps_eff * dataset->Np / dataset->lambda);
   //circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
   cv::circle(planes[0], center, naRadius ,cv::Scalar(1.0), -1, 8, 0);
   
   // FFTshift the pupil so it is consistant with object FT   
   //showImg(planes[0]);
   tmpMat1 = fftShift(planes[0]);
   planes[0] = tmpMat1;
   
   merge(planes, 2, dataset->pupil);
   merge(planes, 2, dataset->pupilSupport);
   //planes[1] = planes[0].clone();
   
   for (int16_t itr = 1; itr <= dataset->itrCount; itr++)
   {
      for (int16_t imgIdx = 1; imgIdx <= dataset->ledCount; imgIdx++)
      {
      int16_t ledNum = dataset->sortedIndicies.at(imgIdx);
      //cout<<ledNum<<endl;
      
      FPMimg * currImg;
      currImg = & dataset->imageStack.at(ledNum);
     
      // Update Fourier space, multply by pupil (P * O)
      currImg->Objfcrop = fftShift(fftShift(dataset->objF)(cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np)));
      
      complexMultiply(currImg->Objfcrop, dataset->pupil, currImg->ObjfcropP);
      if (runDebug)
      {
         showImgMag(dataset->objF,"objF");
         
         // seems to work
         showImgMag(currImg->Objfcrop,"Objfcrop");
         //printComplexPixelValue(currImg->Objfcrop,"Objfcrop",debugPxVal); 
         //maxComplexReal(currImg->Objfcrop,"Objfcrop");
         
         // seems to work
         showImgMag(currImg->ObjfcropP,"ObjfcropP");
         //printComplexPixelValue(currImg->ObjfcropP,"ObjfcropP",debugPxVal);
         //maxComplexReal(currImg->ObjfcropP,"ObjfcropP");
      }
      
      dft(currImg->ObjfcropP, currImg->ObjcropP, DFT_INVERSE | DFT_COMPLEX_OUTPUT | DFT_SCALE); // Real-space of object
      if(runDebug)
      {
           showImgMag(currImg->ObjcropP,"ObjcropP");
           printComplexPixelValue(currImg->ObjcropP,"ObjcropP",debugPxVal);
           maxComplexReal(currImg->ObjcropP,"ObjcropP");
           }
      
      // Replace Amplitude portion
      currImg->Image.convertTo(tmpMat1,CV_64FC1);
      planes[0] = fftShift(tmpMat1); // Need to FFTShift for proper updating
      planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
      cv::merge(planes,2,tmpMat1);
      cv::sqrt(tmpMat1,tmpMat1); // Works because tmpMat is real-valued (complex portion is zero)
      if(runDebug)
        showImgMag(tmpMat1,"Replaced Amplitude, before");
      complexMultiply(tmpMat1, currImg->ObjcropP,tmpMat2); //
      tmpMat3 = tmpMat2.clone(); // Initializes tmpMat3
      
      complexAbs(currImg->ObjcropP + dataset->eps, tmpMat3);
      complexDivide(tmpMat2, tmpMat3, tmpMat1); 
      dft(tmpMat1,currImg->Objfup, DFT_COMPLEX_OUTPUT);
      
      if(runDebug)
      {
           showImgMag(currImg->Objfup,"Objfup");
           printComplexPixelValue(currImg->Objfup,"Objfup",debugPxVal);
           maxComplexReal(currImg->Objfup,"Objfup");
        }
      
      ///////// Alternating Projection Method - Object ///////////
      // MATLAB: Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) = Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) + abs(Pupil).*conj(Pupil).*(Objfup-ObjfcropP)/max(abs(Pupil(:)))./(abs(Pupil).^2+delta2);
      
      // Absolute Value of Pupil
      complexAbs(dataset->pupil,pupilAbs);
      
      if (runDebug)
         showImgMag(pupilAbs,"Pupil Abs");
      
      // Get max of pupil
      cv::minMaxLoc(pupilAbs, &q, &pupilMax);
      
      // Get complex conj of pupil
      complexConj(dataset->pupil, pupilConj);
      if (runDebug)
         showImgMag(pupilConj,"Pupil Conj");

      complexMultiply(pupilAbs, pupilConj, tmpMat1);
      if (runDebug)
      {
         showImgMag(tmpMat1,"pupilAbs * pupilConj");
         maxComplexReal(tmpMat1,"pupilAbs * pupilConj");
         maxComplexReal(currImg->Objfup ,"currImg->Objfup");
         maxComplexReal(currImg->ObjfcropP ,"currImg->ObjfcropP");
      }
      
      //maxComplexReal(currImg->Objfup ,"currImg->Objfup");
      //maxComplexReal(currImg->ObjfcropP ,"currImg->ObjfcropP");
      Mat tmp = currImg->Objfup - currImg->ObjfcropP;
      //showImgMag(tmp,"Data and value difference");
      //showImgMag(tmpMat1,"tmpMat1");
      //maxComplexReal(tmpMat1,"difference");
      complexMultiply(tmpMat1, tmp, tmpMat2); // This is the numerator of the update function
      if (runDebug)
         showImgMag(tmpMat2,"Obj Update Numerator");
      
      maxComplexReal(tmpMat2,"Obj Update Numerator");
      
      complexMultiply(pupilAbs, pupilAbs, tmpMat3); // Pupil Magnitude Squared
      if (runDebug)
         showImgMag(tmpMat3,"Pupil Mag Squared");
      
      // Update incriment for object
      complexDivide(tmpMat2, pupilMax * (tmpMat3 + dataset->delta1), tmpMat1);
      tmpMat2 = fftShift(tmpMat1);
      tmpMat1 = tmpMat2.clone();
      

      if (runDebug)
         showImgMag(tmpMat1,"Object Update Incriment");
      
      // Update fourier space of object
     // tmpMat3 = fftShift(tmpMat1 + dataset->objF(cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np)));
      /*
      if (runDebug)
         showImgMag(tmpMat1,"Object Update Value");
      */
      
      tmpMat2 = fftShift(dataset->objF);
      //showImgMag(tmpMat2,"Updated Image FT (shifted)");
      
      tmpMat3 = cv::Mat(tmpMat2, cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np));
      //showImgMag(tmpMat3,"Updated Image FT ROI (shifted)");
      
      //showImgMag(tmpMat1,"Object Update Incriment");
      
      tmpMat1 = tmpMat3 + tmpMat1;
      //showImgMag(tmpMat1,"Updated Image FT ROI Sum (shifted)");
      
      
      // Replace the region in objF
      tmpMat1.copyTo(cv::Mat(tmpMat2, cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np)));
      dataset->objF = fftShift(tmpMat2);
       
      //showImgMag(tmpMat2,"Updated Image (shifted)");
     

      //if (runDebug)
      //showImgMag(dataset->objF,"Obj Full Fourier Space");
      /////// Alternating Projection Method - Pupil ///////////
      //  MATLAB:  Pupil = Pupil + abs(Objfcrop).*conj(Objfcrop).*(Objfup-ObjfcropP)/max(abs(Objf(:)))./(abs(Objfcrop).^2+delta1).*Pupilsupport;
            
      // Absolute Value of Object F.T.
      complexAbs(currImg->Objfcrop, objfcrop_abs);
      if (runDebug)
         showImgMag(objfcrop_abs,"Object Full Fourier Space Abs");
      
      // Get max of object
      cv::minMaxLoc(objfcrop_abs, &p, &objf_max);
      
      // Get complex conj of object

      complexConj(currImg->Objfcrop, objfcrop_conj);
      if (runDebug)
         showImgMag(objfcrop_conj,"Object Conj");

      complexMultiply(objfcrop_abs, objfcrop_conj, tmpMat1);
      if (runDebug)
      {
         showImgMag(tmpMat1,"Object Conj and abs product");
         showImgMag(currImg->Objfup - currImg->ObjfcropP,"Update difference");
         }
      complexMultiply(tmpMat1, (currImg->Objfup - currImg->ObjfcropP), tmpMat2); // This is the numerator of the update function
      
      /*
      complex_imwrite("tmpMat1.tiff",tmpMat1);
      complex_imwrite("tmpMat2.tiff",tmpMat1);
      complex_imwrite("currImg->Objfup.tiff",currImg->Objfup );
      complex_imwrite("currImg->ObjfcropP.tiff",currImg->ObjfcropP );
      */
      
      if (runDebug)
         showImgMag(tmpMat2,"Pupil update numerator");
            
      complexMultiply(objfcrop_abs, objfcrop_abs, tmpMat3); // Object Magnitude Squared
      if (runDebug)
         showImgMag(tmpMat3,"Object Intensity");
      
      // Update incriment for pupil
      complexDivide(tmpMat2, objf_max * (tmpMat3 + dataset->delta2), tmpMat1);
      complexMultiply(tmpMat1, dataset->pupilSupport, tmpMat1); // Pupil Support
      if (runDebug)
         showImgMag(tmpMat1,"Pupil Update Incriment");

      // Update fourier space of pupil
      //dataset->pupil = dataset->pupil + tmpMat1;
      if (runDebug)
         showImgMag(dataset->pupil,"Pupil Update");
      
      // Pupil = Pupil + abs(Objfcrop).*conj(Objfcrop).*(Objfup-ObjfcropP)/max(abs(Objf(:)))./(abs(Objfcrop).^2+delta1).*Pupilsupport;
      if (runDebug)
         cout << "LED " << ledNum << " completed!" << endl;
         cv::destroyAllWindows();
      }
      cout<<"Iteration " << itr << " Completed!" <<endl;

      dft(dataset->objF ,dataset->objCrop, DFT_INVERSE | DFT_COMPLEX_OUTPUT | DFT_SCALE);
      showImgMag(dataset->objCrop, "Result");

   }  
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
   mDataset.delta2 = 1000;
   
   /* TESTING COMPLEX MAT FUNCTIONS
   Mat testMat1;
   Mat testMat2;
   
   Mat test1[] = {Mat::ones(3, 3, CV_32F)*1,Mat::ones(3, 3, CV_32F)*2};
   Mat test2[] = {Mat::ones(3, 3, CV_32F)*3,Mat::ones(3, 3, CV_32F)*4};
   
   
   Mat test3[] = {Mat::ones(3, 3, CV_32F)*0,Mat::ones(3, 3, CV_32F)*0};
   cv::merge(test1,2,testMat1);
   cv::merge(test2,2,testMat2);
   
   Mat outputMat;
   
   //complexDivide(testMat1,testMat2,outputMat);
   complexConj(testMat1,outputMat);
   
   cv::split(outputMat,test3);
   cv::split(testMat2,test2);
   
   cout << "Input 1 (real) = " << endl << " "  << test1[0]<< ",  (imag) = " << endl << " "  << test1[1] << endl << endl;
   cout << "Input 2 (real) = " << endl << " "  << test2[0] << ",   (imag) = " << endl << " "  << test2[1] << endl << endl;
   cout << "Output (real) = " << endl << " "  << test3[0] <<",   (imag) = " << endl << " "  << test3[1] << endl << endl;
   
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
