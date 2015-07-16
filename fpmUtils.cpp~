/*
fpmUtils.cpp
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
        int16_t               itrCount = 6;            // Iteration Count
        
        float                 eps = 0.0000000001;
};

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
   fileObj2  << ";"<<std::endl;
   
   for (int16_t imgIdx = 0; imgIdx < dataset->ledCount; imgIdx++)
   {
      fileObj2 << dataset->sortedIndicies.at(imgIdx) << "," ;
   }
   fileObj2 << ";" << std::endl;
   
   fileObj2.close();

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
}

void printComplexPixelValue(cv::Mat& m, string label, int16_t pxNum)
{
      Mat planes[] = {Mat::zeros(m.rows, m.cols, m.type()),Mat::zeros(m.rows, m.cols, m.type())};
      split(m,planes);
      cout << "Pixel " << pxNum <<"x"<<pxNum<<" of " <<label << " is: " << planes[0].at<double>(pxNum,pxNum) << " + " << planes[1].at<double>(pxNum,pxNum) << "i " << endl;
      
}
