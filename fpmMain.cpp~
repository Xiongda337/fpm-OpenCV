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

class FPMimg{
  public:
        cv::Mat Image;
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
        int16_t               itrCount = 20;            // Iteration Count
        
        float                 eps = 0.0000000001;
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
          		
          		currentImage.cropYStart = (int16_t)round(dataset->Nlarge/2) + 1 + currentImage.pupilShiftX - (int16_t)round(dataset->Ncrop/2);
          		currentImage.cropYEnd = (int16_t)round(dataset->Nlarge/2) + 1 + currentImage.pupilShiftX + (int16_t)round(dataset->Ncrop/2) - 1;
          		
          		currentImage.cropXStart = (int16_t)round(dataset->Mlarge/2) + 1 + currentImage.pupilShiftY - (int16_t)round(dataset->Ncrop/2);
          		currentImage.cropXEnd = (int16_t)round(dataset->Mlarge/2) + 1 + currentImage.pupilShiftY + (int16_t)round(dataset->Ncrop/2) - 1;
          		
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
    //std::cout<< "Starting Inversion..." << std::endl;
    cv::Mat twiceM = cv::Mat(m.rows * 2, m.cols * 2,CV_32FC1);

    std::vector<cv::Mat> comp;
    cv::split(m,comp);

    cv::Mat real = comp[0];
    cv::Mat imag = comp[1];

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            twiceM.at<float>(i,j) = real.at<float>(i,j);
            twiceM.at<float>(i,j + m.cols) = imag.at<float>(i,j);
            twiceM.at<float>(i + m.rows,j) = -imag.at<float>(i,j);
            twiceM.at<float>(i + m.rows,j + m.cols) = real.at<float>(i,j);
        }
    }

    cv::Mat twiceInv;
    cv::invert(twiceM,twiceInv);

    inverse = cv::Mat(m.cols,m.rows,m.type());

    for(int i=0; i<inverse.rows; i++)
    {
        for(int j=0; j<inverse.cols; j++)
        {
            float re = twiceInv.at<float>(i,j);
            float im = twiceInv.at<float>(i,j + inverse.cols);
            cv::Vec2f val(re,im);
            inverse.at<cv::Vec2f>(i,j) = val;
        }
    }
    //std::cout<< "Inversion Done!" << std::endl;
}

void complexConj(const cv::Mat& m, cv::Mat& output)
{
    Mat planes[] = {Mat::zeros(m.rows, m.cols, CV_32F),Mat::zeros(m.rows, m.cols, CV_32F)};
    cv::split(m, planes);
    planes[1] = -1.0 * planes[1];
    cv::merge(planes, 2, output);
}

void complexAbs(const cv::Mat& m, cv::Mat& output)
{
    Mat planes[] = {Mat::zeros(m.rows, m.cols, CV_32F),Mat::zeros(m.rows, m.cols, CV_32F)};
    cv::pow(m,2,m);
    cv::split(m,planes);
    cv::sqrt(planes[0]+planes[1], output);
}

void complexMultiply(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
{
   Mat outputPlanes[] = {Mat::zeros(m1.rows, m1.cols, CV_32F),Mat::zeros(m1.rows, m1.cols, CV_32F)};
   Mat tmpMat;
   std::vector<cv::Mat> comp1;
   std::vector<cv::Mat> comp2;
   cv::split(m1,comp1);
   cv::split(m2,comp2);
   
   // (a+bi) * (c+di) = ac - bd + (ad+bc) * i
   // Real Part
   cv::multiply(comp1[0], comp2[0], tmpMat);
   outputPlanes[0] = tmpMat;
   cv::multiply(comp1[1], comp2[1], tmpMat);
   outputPlanes[0] = outputPlanes[0] - tmpMat;
   
   // Imaginary Part
   cv::multiply(comp1[0], comp2[1], tmpMat);
   outputPlanes[0] = tmpMat;
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

void run(FPM_Dataset * dataset)
{
   // Make dummy pointers to save space
   Mat * objF = &dataset->objF;
   
   // Initialize FT of reconstructed object with center led image
   Mat complexI;
   Mat planes[] = {Mat_<float>(dataset->imageStack.at(dataset->sortedIndicies.at(1)).Image), Mat::zeros(dataset->Np,dataset->Np, CV_32F)};
   merge(planes, 2, complexI);      
   dft(complexI,dataset->objF);
   copyMakeBorder(dataset->objF, dataset->objF, (dataset->Nlarge-dataset->Np)/2,(dataset->Nlarge-dataset->Np)/2,(dataset->Mlarge-dataset->Np)/2,(dataset->Mlarge-dataset->Np)/2, BORDER_CONSTANT, Scalar::all(0));
   
   // Initialize pupil function
   planes[0] = Mat::zeros(dataset->Np,dataset->Np, CV_32F);
   planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_32F);

   cv::Point center(cvRound(dataset->Np/2),cvRound(dataset->Np/2));
   int16_t naRadius = (int16_t) ceil(dataset->objectiveNA * dataset->ps_eff * dataset->Np / dataset->lambda);
   //circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
   cv::circle(planes[0], center, naRadius ,cv::Scalar(1.0), -1, 8, 0);
   
   Mat tmp;
   // FFTshift the pupil so it is consistant with object FT   
   //showImg(planes[0]);
   tmp = fftShift(planes[0]);
   planes[0] = tmp;
   //showImg(tmp);
   merge(planes, 2, dataset->pupil);
   
   for (int16_t itr = 1; itr <= dataset->itrCount; itr++)
   {
      for (int16_t imgIdx = 1; imgIdx <= dataset->ledCount; imgIdx++)
      {
      int16_t ledNum = dataset->sortedIndicies.at(imgIdx);
      //cout<<ledNum<<endl;
      
      FPMimg * currImg;
      currImg = & dataset->imageStack.at(ledNum);
      /*
        cv::Mat ObjfcropP;
        cv::Mat ObjcropP;
        cv::Mat Objfup;
      */
      
     complexMultiply(dataset->objF(cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np)), dataset->pupil, currImg->ObjfcropP);
      
      dft(currImg->ObjfcropP, currImg->ObjcropP, DFT_INVERSE); // ifft
      
      Mat tmpMat;
      Mat tmpMat2;
      currImg->Image.convertTo(tmpMat,CV_32FC1);
      planes[0] = tmpMat;
      planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_32F);
      cv::merge(planes,2,tmpMat);
      cv::sqrt(tmpMat,tmpMat); // Works because this is real-values (complex portion is zero)
      
      //cv::mulSpectrums(tmpMat,currImg->ObjcropP,tmpMat,0,false); // Complex Matrix multiplication
      complexMultiply(tmpMat,currImg->ObjcropP,tmpMat);
      complexInverse(cv::abs(currImg->ObjcropP+dataset->eps), tmpMat2); // Invert to make the next line a division
      
      cv::mulSpectrums(tmpMat,tmpMat2,tmpMat,0,false); // Complex Matrix multiplication
      dft(tmpMat,currImg->Objfup);
      
     // currImg->ObjcropP;
      //currImg->Objfup;
      
      // Alternating Projection Method
      
      // Absolute Value of Pupil
      Mat pupilAbs;
      complexAbs(dataset->pupil,pupilAbs);
      
      // Get max of pupil
      double q, pupilMax;
      cv::minMaxLoc(pupilAbs, &q, &pupilMax);
      
      // showImg(pupilAbs); // Works
      
      //cv::mulSpectrums(complexAbs(dataset->pupil),complexConj(dataset->pupil),tmpMat,0,false); // Complex Matrix multiplication
      Mat pupilConj;
      complexConj(dataset->pupil, pupilConj);
      
       
      //cv::split(pupilConj, planes);
      //showImg(planes[1]);

      complexMultiply(pupilAbs, pupilConj, tmpMat);
      complexInverse(cv::abs(currImg->ObjcropP+dataset->eps), tmpMat2);
      
      // Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) = Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) + abs(Pupil).*conj(Pupil).*(Objfup-ObjfcropP)/max(abs(Pupil(:)))./(abs(Pupil).^2+delta2);      
      
      // Pupil = Pupil + abs(Objfcrop).*conj(Objfcrop).*(Objfup-ObjfcropP)/max(abs(Objf(:)))./(abs(Objfcrop).^2+delta1).*Pupilsupport;
      cout<<"Iteration"<<endl;
      } 
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
   mDataset.delta1 = 1000;
   mDataset.delta2 = 5;
   
   loadDataset(&mDataset);
   
   run(&mDataset);
   
   //saveFullDataset(&mDataset, "tmpDataset/");
   /*
   showImg(mDataset.imageStack.at(249).Image);
   imwrite("tmp.tiff",mDataset.imageStack.at(249).Image);
  // fpmBackgroundSubtraction(&mDataset);
  */
   
}
