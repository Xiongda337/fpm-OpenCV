/*
fpmMain.cpp
*/
#include <time.h>
///#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <vector>
#include "cvComplex.h"
#include "fpmMain.h"
#include "include/json.h"

//#include "include/rapidjson"
#include "include/domeHoleCoordinates.h"

using namespace std;
using namespace cv;

#define FILENAME_LENGTH 129
#define FILE_HOLENUM_DIGITS 4

string filePrefix = "iLED_";

// Debug flags
bool runDebug = false;
bool loadImgDebug = false;

int16_t loadFPMDataset(FPM_Dataset *dataset) {
  DIR *dir;
  struct dirent *ent;
  Mat fullImg;
  Mat fullImgComplex;
  FPMimg tmpFPMimg;
  tmpFPMimg.Image = Mat::zeros(dataset->Np, dataset->Np, CV_8UC1);

  double angle = dataset->arrayRotation;
  Mat rotationMatrixZ = (Mat_<double>(3,3) << cos (angle*M_PI/180), -sin (angle*M_PI/180), 0,sin (angle*M_PI/180), cos (angle*M_PI/180), 0, 0, 0, 1);

  clock_t t1, t2;
  if (loadImgDebug) {
    t1 = clock();
  }

  // Initialize array of image objects, since we don't know exactly what order
  // imread will read in images. First (0th) element is a dummy so we can access
  // these using the led # directly
  for (int16_t ledIdx = 0; ledIdx <= dataset->ledCount; ledIdx++) {
    dataset->imageStack.push_back(tmpFPMimg);
    dataset->illuminationNAList.push_back(99.0);
    dataset->NALedPatternStackX.push_back(-1.0);
    dataset->NALedPatternStackY.push_back(-1.0);
  }

  if ((dir = opendir(dataset->datasetRoot.c_str())) != NULL) {
    int16_t num_images = 0;
    std::cout << "Loading Images..." << std::endl;
    while ((ent = readdir(dir)) != NULL) {
      string fileName = ent->d_name;
      /* Get data from file name, if name is right format.*/
      if (fileName.compare(".") != 0 && fileName.compare("..") != 0 &&
          (strcmp(dataset->fileExtension.c_str(),
                  &(ent->d_name[strlen(ent->d_name) -
                                dataset->fileExtension.length()])) == 0) &&(
                              fileName.find(dataset->filePrefix) == 0))
                              {
        string holeNum = fileName.substr(fileName.find(dataset->filePrefix) +
                                             dataset->filePrefix.length(),
                                         FILE_HOLENUM_DIGITS);
       FPMimg currentImage;
       currentImage.led_num = atoi(holeNum.c_str());

       Mat holeCoordinates = (Mat_<double>(1,3) << domeHoleCoordinates[currentImage.led_num - 1][0], domeHoleCoordinates[currentImage.led_num - 1][1], domeHoleCoordinates[currentImage.led_num - 1][2]);
       holeCoordinates = rotationMatrixZ * holeCoordinates.t();
       currentImage.sinTheta_x =
           sin(atan2(holeCoordinates.at<double>(0, 0),
                     holeCoordinates.at<double>(2, 0)));
       currentImage.sinTheta_y =
       sin(atan2(holeCoordinates.at<double>(1, 0),
                 holeCoordinates.at<double>(2, 0)));
       currentImage.illumination_na =
                     sqrt(currentImage.sinTheta_x * currentImage.sinTheta_x +
                          currentImage.sinTheta_y * currentImage.sinTheta_y);
  std::cout <<"NA:"<<sqrt(currentImage.sinTheta_x * currentImage.sinTheta_x + currentImage.sinTheta_y * currentImage.sinTheta_y)<<endl;
         if (sqrt(currentImage.illumination_na<dataset->maxIlluminationNA))
         {


        if (dataset->color) {
          fullImg = imread(dataset->datasetRoot + fileName,
                           CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR);
          Mat channels[3];
          split(fullImg, channels);
          cvtColor(fullImg, fullImg, CV_RGB2GRAY);
          fullImg = channels[2]; // Green Channel
        //  cout << fullImg.depth() << std::endl;
        //  cout << fullImg.channels() << std::endl;
        } else {

          fullImg =
              imread(dataset->datasetRoot + fileName, CV_LOAD_IMAGE_ANYDEPTH);
        }

        // Populate fields of FPMImage class
        currentImage.Image = fullImg(cv::Rect(dataset->cropX, dataset->cropY,
                                              dataset->Np, dataset->Np))
                                 .clone();

        cv::Scalar bk1 = cv::mean(fullImg(cv::Rect(
            dataset->bk1cropX, dataset->bk1cropY, dataset->Np, dataset->Np)));
        cv::Scalar bk2 = cv::mean(fullImg(cv::Rect(
            dataset->bk2cropX, dataset->bk2cropY, dataset->Np, dataset->Np)));

        double bg_val = ((double)bk2[0] + (double)bk1[0]) / 2;
        if (bg_val > dataset->bgThreshold)
          bg_val = dataset->bgThreshold;

        currentImage.bg_val = (int16_t)round(bg_val);

        // Perform Background Subtraction
        cv::subtract(currentImage.Image, cv::Scalar(currentImage.bg_val, 0, 0),
                     currentImage.Image);

        currentImage.uled = currentImage.sinTheta_x / dataset->lambda;
        currentImage.vled = currentImage.sinTheta_y / dataset->lambda;

        // Fourier shift of off-axis led
        currentImage.idx_u = (int16_t)round(currentImage.uled / dataset->du);
        currentImage.idx_v = (int16_t)round(currentImage.vled / dataset->du);
        // Pixel shift of off-axis led
        currentImage.pupilShiftX = (int16_t)round(
            currentImage.sinTheta_x / dataset->lambda * dataset->ps *
            dataset->Nlarge); // Deal with MATLAB indexing
        currentImage.pupilShiftY = (int16_t)round(
            currentImage.sinTheta_y / dataset->lambda * dataset->ps *
            dataset->Mlarge); // Deal with MATLAB indexing
                              // Region to crop in Fourier Domain
        currentImage.cropXStart = (int16_t)round(dataset->Nlarge / 2) +
                                  currentImage.pupilShiftX -
                                  (int16_t)round(dataset->Ncrop / 2);
        currentImage.cropXEnd = (int16_t)round(dataset->Nlarge / 2) +
                                currentImage.pupilShiftX +
                                (int16_t)round(dataset->Ncrop / 2) - 1;
        currentImage.cropYStart = (int16_t)round(dataset->Mlarge / 2) +
                                  currentImage.pupilShiftY -
                                  (int16_t)round(dataset->Ncrop / 2);
        currentImage.cropYEnd = (int16_t)round(dataset->Mlarge / 2) +
                                currentImage.pupilShiftY +
                                (int16_t)round(dataset->Ncrop / 2) - 1;

        // Assign Image object to FPMDataset class
        dataset->imageStack.at(currentImage.led_num) = currentImage;
        dataset->illuminationNAList.at(currentImage.led_num) =
            currentImage.illumination_na;
        dataset->NALedPatternStackX.at(currentImage.led_num) =
            currentImage.sinTheta_x;
        dataset->NALedPatternStackY.at(currentImage.led_num) =
            currentImage.sinTheta_y;

        num_images++;
        std::cout << "Loaded: " << fileName
                  << ", LED # is: " << currentImage.led_num << std::endl;
        if (loadImgDebug) {
          std::cout << "   sintheta_x is: " << currentImage.sinTheta_x
                    << ", sintheta_y is: " << currentImage.sinTheta_y
                    << std::endl;
          std::cout << "   Image Size is: " << currentImage.Image.rows << " x "
                    << currentImage.Image.cols << std::endl;
          std::cout << "   cartx is: "
                    << domeHoleCoordinates[currentImage.led_num - 1][0]
                    << std::endl;
          std::cout << "   carty is: "
                    << domeHoleCoordinates[currentImage.led_num - 1][1]
                    << std::endl;
          std::cout << "   cartz is: "
                    << domeHoleCoordinates[currentImage.led_num - 1][2]
                    << std::endl;
          std::cout << "   atan(cartx) : "
                    << atan(domeHoleCoordinates[currentImage.led_num - 1][0] /
                            domeHoleCoordinates[currentImage.led_num - 1][2])
                    << std::endl;
          std::cout << "   atan(carty) : "
                    << atan(domeHoleCoordinates[currentImage.led_num - 1][1] /
                            domeHoleCoordinates[currentImage.led_num - 1][2])
                    << std::endl;
          std::cout << "   sin(atan(cartx)) : "
                    << sin(atan(
                           domeHoleCoordinates[currentImage.led_num - 1][0] /
                           domeHoleCoordinates[currentImage.led_num - 1][2]))
                    << std::endl;
          std::cout << "   sin(atan(carty)) : "
                    << sin(atan(
                           domeHoleCoordinates[currentImage.led_num - 1][1] /
                           domeHoleCoordinates[currentImage.led_num - 1][2]))
                    << std::endl;
          std::cout << "   uled : " << currentImage.uled << std::endl;
          std::cout << "   vled : " << currentImage.vled << std::endl;
          std::cout << "   idx_u : " << currentImage.idx_u << std::endl;
          std::cout << "   idx_v : " << currentImage.idx_v << std::endl;
          std::cout << "   pupilShiftX : " << currentImage.pupilShiftX
                    << std::endl;
          std::cout << "   pupilShiftY : " << currentImage.pupilShiftY
                    << std::endl;
          std::cout << "   cropXStart : " << currentImage.cropXStart
                    << std::endl;
          std::cout << "   cropXEnd : " << currentImage.cropXEnd << std::endl;
          std::cout << "   cropYStart : " << currentImage.cropYStart
                    << std::endl;
          std::cout << "   cropYEnd : " << currentImage.cropYEnd << std::endl;
          std::cout << "   illumination na: " << currentImage.illumination_na
                    << std::endl;
          std::cout << std::endl << std::endl;
        }
      }else
        std::cout << "Skipped LED# " << holeNum <<std::endl;
      }
      dataset->ledUsedCount = num_images;
    }
    closedir(dir);
    if (num_images <= 0) {
      std::cout << "ERROR - No images found in given directory." << std::endl;
      return -1;
    }

    // Sort the Images into the correct order
    int16_t indexIncr = 1;
    for (auto i : sort_indexes(dataset->illuminationNAList)) {
      if (indexIncr <= dataset->ledUsedCount)
      {
        dataset->sortedIndicies.push_back(i);
        dataset->sortedNALedPatternStackX.push_back(
            dataset->NALedPatternStackX[i]);
        dataset->sortedNALedPatternStackY.push_back(
            dataset->NALedPatternStackY[i]);
        indexIncr++;
      }
    }
    if (loadImgDebug) {
      t2 = clock();
      float diff(((float)t2 - (float)t1) / CLOCKS_PER_SEC);
      cout << "Image loading Completed (Time: " << diff << " sec)" << endl;
      return num_images;
    }
  return 1;
  } else {
    /* could not open directory */
    std::cout << "ERROR: Could not Open Directory.\n";
    return -1;
  }
}

void runFPM(FPM_Dataset *dataset) {

  clock_t t1, t2, t3, t4;
  t3 = clock();
  // Make dummy pointers to save space
  Mat *objF = &dataset->objF;

  // Initilize Matricies
  Mat tmpMat1, tmpMat2, tmpMat3;
  Mat objF_centered;
  Mat complexI, pupilAbs, pupilConj, objfcrop_abs, objfcrop_conj;
  Mat Objfcrop_abs;
  Mat Objfcrop_abs_sq;
  Mat Objf_abs;
  Mat Objfcrop_conj;
  Mat Objfcrop_abs_conj;
  Mat planes[] = {Mat::zeros(dataset->Np, dataset->Np, CV_64F),
                  Mat::zeros(dataset->Np, dataset->Np, CV_64F)};
  Mat objectAmp = Mat::zeros(dataset->Np, dataset->Np, CV_64FC2);
  Mat pupil_abs;
  Mat pupil_abs_sq;
  Mat pupil_conj;
  Mat numerator;
  Mat denomSum;
  double q, pupilMax, p, objf_max, Objf_abs_max;
  FPMimg *currImg;

  // Initialize pupil function
  planes[0] = Mat::zeros(dataset->Np, dataset->Np, CV_64F);
  planes[1] = Mat::zeros(dataset->Np, dataset->Np, CV_64F);
  cv::Point center(cvRound(dataset->Np / 2), cvRound(dataset->Np / 2));
  int16_t naRadius = (int16_t)ceil(dataset->objectiveNA * dataset->ps_eff *
                                   dataset->Np / dataset->lambda);
  cv::circle(planes[0], center, naRadius, cv::Scalar(1.0), -1, 8, 0);

  // FFTshift the pupil so it is consistant with object FT
  fftShift(planes[0], planes[0]);

  merge(planes, 2, dataset->pupil);
  dataset->pupilSupport = dataset->pupil.clone();

  // Initialize FT of reconstructed object with center led image
  planes[0] =
      Mat_<double>(dataset->imageStack.at(dataset->sortedIndicies.at(1)).Image);
  cv::sqrt(planes[0], planes[0]); // Convert to amplitude
  planes[1] = Mat::zeros(dataset->Np, dataset->Np, CV_64F);
  merge(planes, 2, complexI);

  fft2(complexI, complexI);
  complexMultiply(complexI, dataset->pupilSupport, complexI);
  fftShift(complexI, complexI); // Shift to center

  planes[0] = Mat::zeros(dataset->Nlarge, dataset->Mlarge, CV_64F);
  planes[1] = Mat::zeros(dataset->Nlarge, dataset->Mlarge, CV_64F);
  merge(planes, 2, dataset->objF);

  complexI.copyTo(
      cv::Mat(dataset->objF, cv::Rect((int16_t)round(dataset->Mlarge / 2) -
                                          (int16_t)round(dataset->Ncrop / 2),
                                      (int16_t)round(dataset->Mlarge / 2) -
                                          (int16_t)round(dataset->Ncrop / 2),
                                      dataset->Np, dataset->Np)));

  // Shift to un-fftshifted position
  fftShift(dataset->objF, dataset->objF);

  for (int16_t itr = 1; itr <= dataset->itrCount; itr++)
  {
    t1 = clock();
    for (int16_t imgIdx = 0; imgIdx < dataset->ledUsedCount; imgIdx++) //
    {
      int16_t ledNum = dataset->sortedIndicies.at(imgIdx);

      if (runDebug)
        cout << "Starting LED# " << ledNum << endl;

      currImg = &dataset->imageStack.at(ledNum);

      // Update Fourier space, multply by pupil (P * O)
      fftShift(dataset->objF,
               objF_centered); // Shifted Object spectrum (at center)
      fftShift(objF_centered(cv::Rect(currImg->cropXStart, currImg->cropYStart,
                                      dataset->Np, dataset->Np)),
               currImg->Objfcrop); // Take ROI from shifted object spectrum

      complexMultiply(currImg->Objfcrop, dataset->pupil, currImg->ObjfcropP);
      ifft2(currImg->ObjfcropP, currImg->ObjcropP);
      if (runDebug) {
        std::cout << "NEW LED" << std::endl;
        showComplexImg(currImg->Objfcrop, SHOW_COMPLEX_MAG,
                       "currImg->Objfcrop");
        showComplexImg(currImg->ObjfcropP, SHOW_COMPLEX_MAG,
                       "currImg->ObjfcropP");
        showComplexImg(currImg->ObjcropP, SHOW_COMPLEX_COMPONENTS,
                       "currImg->ObjcropP");
      }

      // Replace Amplitude (using pointer iteration)
      for (int i = 0; i < dataset->Np; i++) // loop through y
      {
        const uint16_t *m_i = currImg->Image.ptr<uint16_t>(i); // Input
        double *o_i = objectAmp.ptr<double>(i);                // Output

        for (int j = 0; j < dataset->Np; j++) {
          o_i[j * 2] = sqrt((double)m_i[j]); // Real
          o_i[j * 2 + 1] = 0.0;              // Imaginary
        }
      }
      // Update Onject fourier transform (preserbing phase)
      complexAbs(currImg->ObjcropP + dataset->eps, tmpMat3);
      complexDivide(currImg->ObjcropP, tmpMat3, tmpMat1);
      complexMultiply(tmpMat1, objectAmp, tmpMat3);
      fft2(tmpMat3, currImg->Objfup);

      if (runDebug) {
        showComplexImg(objectAmp, SHOW_COMPLEX_COMPONENTS,
                       "Amplitude of Input Image");
        showComplexImg(tmpMat3, SHOW_COMPLEX_COMPONENTS,
                       "Image with amplitude   replaced");
        showComplexImg(currImg->Objfup, SHOW_COMPLEX_MAG, "currImg->Objfup");
      }

      ///////// Alternating Projection Method - Object ///////////
      // Numerator
      complexAbs(dataset->pupil, pupil_abs);
      complexConj(dataset->pupil, pupil_conj);
      complexMultiply(pupil_abs, pupil_conj, tmpMat1);
      complexMultiply(currImg->Objfup - currImg->ObjfcropP, tmpMat1, numerator);

      // Denominator
      double p;
      double pupil_abs_max;
      cv::minMaxLoc(pupil_abs, &p, &pupil_abs_max);
      complexMultiply(pupil_abs, pupil_abs, pupil_abs_sq);
      denomSum = pupil_abs_sq + dataset->delta2;
      complexDivide(numerator, denomSum * pupil_abs_max, tmpMat2);

      if (runDebug) {
        showComplexImg(numerator, SHOW_COMPLEX_MAG, "Object update Numerator");
        showComplexImg(tmpMat2, SHOW_COMPLEX_MAG, "Object update Denominator");
      }

      fftShift(dataset->objF, objF_centered);

      Mat objF_cropped = cv::Mat(
          objF_centered, cv::Rect(currImg->cropXStart, currImg->cropYStart,
                                  dataset->Np, dataset->Np));
      fftShift(tmpMat2, tmpMat2);
      tmpMat1 = tmpMat2 + objF_cropped;

      if (runDebug) {
        showComplexImg(objF_cropped, SHOW_COMPLEX_MAG,
                       "Origional Object Spectrum to be updated");
        fftShift(tmpMat2, tmpMat2);
        showComplexImg(tmpMat2, SHOW_COMPLEX_MAG,
                       "Object spectrum update incriment");
      }

      // Replace the region in objF
      tmpMat1.copyTo(cv::Mat(objF_centered,
                             cv::Rect(currImg->cropXStart, currImg->cropYStart,
                                      dataset->Np, dataset->Np)));
      fftShift(objF_centered, dataset->objF);

      if (runDebug) {
        fftShift(tmpMat1, tmpMat1);
        showComplexImg(tmpMat1, SHOW_COMPLEX_MAG,
                       "Cropped updated object spectrum");
        showComplexImg(dataset->objF, SHOW_COMPLEX_MAG,
                       "Full updated object spectrum");
      }

      ////// PUPIL UPDATE ///////
      // Numerator
      complexAbs(currImg->Objfcrop, Objfcrop_abs);
      complexAbs(dataset->objF, Objf_abs);
      complexConj(currImg->Objfcrop, Objfcrop_conj);
      complexMultiply(Objfcrop_abs, Objfcrop_conj, tmpMat1);
      complexMultiply(currImg->Objfup - currImg->ObjfcropP, tmpMat1, numerator);

      // Denominator
      cv::minMaxLoc(Objf_abs, &p, &Objf_abs_max);
      complexMultiply(Objfcrop_abs, Objfcrop_abs, Objfcrop_abs_sq);
      denomSum = Objfcrop_abs_sq + dataset->delta1;
      complexDivide(numerator, denomSum * Objf_abs_max, tmpMat2);
      complexMultiply(tmpMat2, dataset->pupilSupport, tmpMat2);

      dataset->pupil += tmpMat2;
    }
    t2 = clock();
    float diff(((float)t2 - (float)t1) / CLOCKS_PER_SEC);
    cout << "Iteration " << itr << " Completed (Time: " << diff << " sec)"
         << endl;
    dft(dataset->objF, dataset->objCrop, DFT_INVERSE | DFT_SCALE);
  }
  // showImgObject((dataset->objCrop), "Object");
  // showImgFourier((dataset->objF),"Object Spectrum");
  // showImgObject(fftShift(dataset->pupil),"Pupil");
  t4 = clock();
  float diff(((float)t4 - (float)t3) / CLOCKS_PER_SEC);
  cout << "FP Processing Completed (Time: " << diff << " sec)" << endl;

  // showComplexImg(dataset->objF, SHOW_COMPLEX_MAG, "Object Spectrum");
  showComplexImg(dataset->objCrop, SHOW_COMPLEX_COMPONENTS, "Object");
  fftShift(dataset->pupil, dataset->pupil);
  showComplexImg(dataset->pupil, SHOW_COMPLEX_COMPONENTS, "Pupil");
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout
        << "ERROR: Not enough input argumants.\n Usage: ./fpmMain dataset.json"
        << std::endl;
    return 0;
  }

  // The dataset object, which contains all images and experimental parameters
  FPM_Dataset mDataset;

  // Load parameters from json file
  Json::Value datasetJson;
  Json::Reader reader;
  ifstream jsonFile(argv[1]);
  reader.parse(jsonFile, datasetJson);

  mDataset.filePrefix = datasetJson.get("filePrefix", "iLED_").asString();
  mDataset.fileExtension = datasetJson.get("fileExtension", ".tif").asString();
  mDataset.Np = datasetJson.get("cropSizeX", 90).asInt();
  mDataset.datasetRoot = datasetJson.get("datasetRoot", ".").asString();
  mDataset.pixelSize = datasetJson.get("pixelSize", 6.5).asDouble();
  mDataset.objectiveMag = datasetJson.get("objectiveMat", 8).asDouble();
  mDataset.objectiveNA = datasetJson.get("objectiveNA", 0.2).asDouble();
  mDataset.maxIlluminationNA =
      datasetJson.get("maxIlluminationNA", 0.7604).asDouble();
  mDataset.color = datasetJson.get("isColor", false).asBool();
  mDataset.centerLED = datasetJson.get("centerLED", 249).asInt();
  mDataset.lambda = datasetJson.get("lambda", 0.5).asDouble();
  mDataset.ps_eff = mDataset.pixelSize / (float)mDataset.objectiveMag;
  mDataset.du = (1 / mDataset.ps_eff) / (float)mDataset.Np;
  mDataset.leadingZeros = datasetJson.get("leadingZeros", false).asBool();
  mDataset.cropX = datasetJson.get("cropX", 1).asInt();
  mDataset.cropY = datasetJson.get("cropY", 1).asInt();
  mDataset.arrayRotation = datasetJson.get("arrayRotation", 0).asInt();
  mDataset.bk1cropX = datasetJson.get("bk1cropX", 1).asInt();
  mDataset.bk1cropY = datasetJson.get("bk1cropY", 1).asInt();
  mDataset.bk2cropX = datasetJson.get("bk2cropX", 1).asInt();
  mDataset.bk2cropY = datasetJson.get("bk2cropY", 1).asInt();
  mDataset.holeNumberDigits = datasetJson.get("holeNumberDigits",4).asInt();

  std::cout << "Dataset Root: " << mDataset.datasetRoot << std::endl;
  char fileName[FILENAME_LENGTH];
  sprintf(fileName, "%s%04d%s", mDataset.filePrefix.c_str(), mDataset.centerLED,
          mDataset.fileExtension.c_str());

  cout << mDataset.datasetRoot + fileName << endl;

  if (loadImgDebug)
  {
    Mat img = imread(mDataset.datasetRoot + fileName, CV_LOAD_IMAGE_ANYDEPTH);
    std::cout << img.size();
    showImg(
    img(cv::Rect(mDataset.cropX, mDataset.cropY, mDataset.Np, mDataset.Np)),
      "Cropped Center Image");
  }
  /*int16_t resImprovementFactor = (int16_t)ceil(
      2 * mDataset.ps_eff *
      (mDataset.maxIlluminationNA + mDataset.objectiveNA) /
      mDataset
          .lambda);*/

  int16_t resImprovementFactor = 5;
  mDataset.bgThreshold = datasetJson.get("bgThresh", 1000).asInt();
  mDataset.Mcrop = mDataset.Np;
  mDataset.Ncrop = mDataset.Np;
  mDataset.Nlarge = mDataset.Ncrop * resImprovementFactor;
  mDataset.Mlarge = mDataset.Mcrop * resImprovementFactor;
  mDataset.ps = mDataset.ps_eff / (float)resImprovementFactor;
  mDataset.delta1 = datasetJson.get("delta1", 5).asInt();
  mDataset.delta2 = datasetJson.get("delta2", 10).asInt();
  mDataset.itrCount = atoi(argv[2]);
  mDataset.ledCount = datasetJson.get("ledCount", 508).asInt();

  // Reserve memory for imageStack
  mDataset.imageStack.reserve(mDataset.ledCount);

  // Load the datastack, and process if we find images
  if(loadFPMDataset(&mDataset) > 0)
    runFPM(&mDataset);
}
