#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <vector>
#include "include/json.h"

#if !defined(FPM_MAIN_H)
#define FPM_MAIN_H 1

using namespace std;
using namespace cv;

class FPMimg{
  public:
        cv::UMat Image;
        cv::UMat Objfcrop;
        cv::UMat ObjfcropP;
        cv::UMat ObjcropP;
        cv::UMat Objfup;
        int led_num;
        double sinTheta_x;
        double sinTheta_y;
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
        std::string           holeCoordinateFileName;
        Json::Value           holeCoordinates;
        std::string           datasetRoot;    // Dataset location, with trailing "\"
        std::string           filePrefix;          // Raw data file header (everything before led #)
        std::string           fileExtension;       // Raw data file extension (e.g. .tif)
        std::vector<FPMimg>   imageStack;
        double                arrayRotation;       // Global Rotaton of array
        uint16_t              darkfieldExpMultiplier;
        int16_t               holeNumberDigits;    // Number of digits in hole number in filename
        uint16_t              ledCount;            // Number of LEDs in system (even if we don't use all of them)
        uint16_t              ledUsedCount;
        float                 pixelSize;           // pixel size in microns
        float                 objectiveMag;        // Objective Magnification
        float                 objectiveNA;         // Objective NA
        float                 maxIlluminationNA;   // Max illumination NA
        float                 lambda;              // wavelength in microns
        bool                  color;               // flag for data acquired on color camera
        bool                  leadingZeros;        // flag for definint whether or not the filenames have leading zeros
        int16_t               centerLED = 249;     // Closest LED to center of Array
        int16_t               cropX;               // X position of crop region start
        int16_t               cropY;               // Y position of crop region start
        int16_t               Np;                  // ROI Size
        int16_t               Np_padded;           // Zero-padded ROI
        int16_t               Mcrop;               // Size of upsampled image (x)
        int16_t               Ncrop;               // Size of upsampled image (y)
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
        cv::UMat               obj;                      // Reconstructed object, real space, full res
        cv::UMat               objCrop;                  // Reconstructed object, real space, cropped
        cv::UMat               objF;                     // Reconstructed object, Fourier space, full res
        cv::UMat               objFCrop;                 // Reconstructed object, Fourier space, cropped
        cv::UMat               pupil;                     // Reconstructed pupil, Fourier Space
        cv::UMat               pupilSupport;             // Binary mask for pupil support, Fourier space
        int16_t               itrCount = 10;            // Iteration Count
        bool                  flipIlluminationX;
        bool                  flipIlluminationY;
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
int16_t loadFPMDataset(FPM_Dataset *dataset);
void runFPM(FPM_Dataset * dataset);

#endif
