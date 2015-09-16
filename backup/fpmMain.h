
#include <time.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <vector>

#if !defined(FPM_MAIN_H)
#define FPM_MAIN_H 1

using namespace std;
using namespace cv;

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
        std::string                datasetRoot;   // Folder to search for images (including slash at end)
        std::string           filePrefix;         // Image file prefix (characters before led #)
        std::string           fileExtension;      // Image file extension
        std::vector<FPMimg>   imageStack;         // List of image objects
        uint16_t              ledCount;           // Number of LEDs (int)
        float                 pixelSize;           // pixel size in microns
        float                 objectiveMag;       // Objective Magnification
        float                 objectiveNA;        // Objective NA
        float                 maxIlluminationNA;  // Illumination NA
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

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

// Function Definitions
int loadDataset(FPM_Dataset *dataset);
void run(FPM_Dataset * dataset);

#endif
