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

//#include "include/rapidjson"
#include "include/domeHoleCoordinates.h"

using namespace std;
using namespace cv;

#define FILENAME_LENGTH 129
#define FILE_HOLENUM_DIGITS 4

string filePrefix = "iLED_";

// Debug flags
bool preprocessDebug = false;
bool runDebug = false;

int loadDataset(FPM_Dataset *dataset) {
	DIR *dir;
	struct dirent *ent;
	Mat fullImg;
	Mat fullImgComplex;
	FPMimg tmpFPMimg;
	tmpFPMimg.Image = Mat::zeros(dataset->Np, dataset->Np, CV_8UC1);
	
	clock_t t1,t2;
	t1=clock();
	
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
      
     t2=clock();
     float diff (((float)t2-(float)t1) / CLOCKS_PER_SEC);
     cout<<"Image loading Completed (Time: " << diff << " sec)"<<endl;
	  return num_images;

	} else {
	  /* could not open directory */
	  std::cout << "ERROR: Could not Load Images.\n";
	  //perror ("");
	  return EXIT_FAILURE;
	}
}

void runFPM(FPM_Dataset * dataset)
{
      
   clock_t t1,t2,t3,t4;
   t3 = clock();
   // Make dummy pointers to save space
   Mat * objF = &dataset->objF;
   
   // Initilize Matricies
   Mat tmpMat1, tmpMat2, tmpMat3;
   Mat objF_centered;
   Mat complexI, pupilAbs, pupilConj, objfcrop_abs, objfcrop_conj;
   Mat Objfcrop_abs; Mat Objfcrop_abs_sq; Mat Objf_abs; Mat Objfcrop_conj; Mat Objfcrop_abs_conj;
   Mat planes[] = {Mat::zeros(dataset->Np,dataset->Np, CV_64F), Mat::zeros(dataset->Np,dataset->Np, CV_64F)};
   Mat objectAmp = Mat::zeros(dataset->Np,dataset->Np, CV_64FC2);
   Mat pupil_abs; Mat pupil_abs_sq; Mat pupil_conj; Mat numerator; Mat denomSum;
   double q, pupilMax, p, objf_max, Objf_abs_max;
   FPMimg * currImg;

   // Initialize pupil function
   planes[0] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
   planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
   cv::Point center(cvRound(dataset->Np/2),cvRound(dataset->Np/2));
   int16_t naRadius = (int16_t) ceil(dataset->objectiveNA * dataset->ps_eff * dataset->Np / dataset->lambda);
   cv::circle(planes[0], center, naRadius ,cv::Scalar(1.0), -1, 8, 0);
   
   // FFTshift the pupil so it is consistant with object FT   
   fftShift(planes[0],planes[0]);
   
   merge(planes, 2, dataset->pupil);
   dataset->pupilSupport = dataset->pupil.clone();
   
   // Initialize FT of reconstructed object with center led image

   planes[0] = Mat_<double>(dataset->imageStack.at(dataset->sortedIndicies.at(1)).Image);
   cv::sqrt(planes[0], planes[0]); // Convert to amplitude
   planes[1] = Mat::zeros(dataset->Np,dataset->Np, CV_64F);
   merge(planes, 2, complexI);
   
   fft2(complexI, complexI);
   complexMultiply(complexI,dataset->pupilSupport,complexI);
   fftShift(complexI,complexI); // Shift to center
   
   planes[0] = Mat::zeros(dataset->Nlarge,dataset->Mlarge, CV_64F);
   planes[1] = Mat::zeros(dataset->Nlarge,dataset->Mlarge, CV_64F);
   merge(planes,2,dataset->objF);
   
   complexI.copyTo(cv::Mat(dataset->objF, cv::Rect((int16_t)round(dataset->Mlarge/2) - (int16_t)round(dataset->Ncrop/2),(int16_t)round(dataset->Mlarge/2) - (int16_t)round(dataset->Ncrop/2),dataset->Np,dataset->Np)));
    
   // Shift to un-fftshifted position
   fftShift(dataset->objF,dataset->objF);


   for (int16_t itr = 1; itr <= dataset->itrCount; itr++)
   {
      t1=clock();
      for (int16_t imgIdx = 1; imgIdx <= dataset->ledCount; imgIdx++) // 
      {
         int16_t ledNum = dataset->sortedIndicies.at(imgIdx);
         if (runDebug)
            cout<< "Starting LED# " <<ledNum<<endl;
         
         currImg = &dataset->imageStack.at(ledNum);
        
         // Update Fourier space, multply by pupil (P * O)
         fftShift(dataset->objF, objF_centered); // Shifted Object spectrum (at center)
         fftShift(objF_centered(cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np, dataset->Np)),currImg->Objfcrop); // Take ROI from shifted object spectrum
         
         complexMultiply(currImg->Objfcrop, dataset->pupil, currImg->ObjfcropP);
         ifft2(currImg->ObjfcropP,currImg->ObjcropP);
         if (runDebug)
         {
            std::cout << "NEW LED" <<std::endl;
            showComplexImg(currImg->Objfcrop, SHOW_COMPLEX_MAG, "currImg->Objfcrop");
            showComplexImg(currImg->ObjfcropP, SHOW_COMPLEX_MAG, "currImg->ObjfcropP");
            showComplexImg(currImg->ObjcropP, SHOW_COMPLEX_COMPONENTS, "currImg->ObjcropP");
         }

	      // Replace Amplitude (using pointer iteration)
			for(int i = 0; i < dataset->Np; i++) // loop through y
			{
			 const uint16_t* m_i = currImg->Image.ptr<uint16_t>(i);  // Input
			 double* o_i = objectAmp.ptr<double>(i);   // Output
			 
			 for(int j = 0; j < dataset->Np; j++)
			 {
				  o_i[j*2] = sqrt((double) m_i[j]); // Real
				  o_i[j*2+1] = 0.0; // Imaginary
			 }
			}
	
         complexAbs(currImg->ObjcropP + dataset->eps, tmpMat3);
         complexDivide(currImg->ObjcropP, tmpMat3, tmpMat1);
         complexMultiply(tmpMat1, objectAmp ,tmpMat3);
         fft2(tmpMat3,currImg->Objfup);
         
         if(runDebug)
         {
              showComplexImg(objectAmp, SHOW_COMPLEX_COMPONENTS,"Amplitude of Input Image");
              showComplexImg(tmpMat3, SHOW_COMPLEX_COMPONENTS,"Image with amplitude   replaced");
              showComplexImg(currImg->Objfup,SHOW_COMPLEX_MAG,"currImg->Objfup");
         }
         
         ///////// Alternating Projection Method - Object ///////////
         // MATLAB: Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) = Objf(cropystart(j):cropyend(j),cropxstart(j):cropxend(j)) + abs(Pupil).*conj(Pupil).*(Objfup-ObjfcropP)/max(abs(Pupil(:)))./(abs(Pupil).^2+delta2);
         
         // Numerator 
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
              showComplexImg(numerator, SHOW_COMPLEX_MAG, "Object update Numerator");
              showComplexImg(tmpMat2, SHOW_COMPLEX_MAG, "Object update Denominator");
         }
        
         fftShift(dataset->objF,objF_centered);
         
         Mat objF_cropped = cv::Mat(objF_centered, cv::Rect(currImg->cropXStart, currImg->cropYStart, dataset->Np, dataset->Np));
         fftShift(tmpMat2,tmpMat2);
         tmpMat1 = tmpMat2 + objF_cropped;
         
         if(runDebug)
         {
              showComplexImg(objF_cropped,SHOW_COMPLEX_MAG,"Origional Object Spectrum to be updated");
              fftShift(tmpMat2,tmpMat2);
              showComplexImg(tmpMat2,SHOW_COMPLEX_MAG,"Object spectrum update incriment");
         }

         // Replace the region in objF
         tmpMat1.copyTo(cv::Mat(objF_centered, cv::Rect(currImg->cropXStart,currImg->cropYStart,dataset->Np,dataset->Np)));
         fftShift(objF_centered,dataset->objF);
         
         if(runDebug)
         {
              fftShift(tmpMat1,tmpMat1);
              showComplexImg(tmpMat1,SHOW_COMPLEX_MAG,"Cropped updated object spectrum");
              showComplexImg(dataset->objF,SHOW_COMPLEX_MAG,"Full updated object spectrum");
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
         complexMultiply(Objfcrop_abs,Objfcrop_abs,Objfcrop_abs_sq);
         denomSum = Objfcrop_abs_sq + dataset->delta1;
         complexDivide(numerator, denomSum * Objf_abs_max, tmpMat2);
         complexMultiply(tmpMat2,dataset->pupilSupport, tmpMat2);
         
         dataset->pupil += tmpMat2; 
         
      }
      t2=clock();
      float diff (((float)t2-(float)t1) / CLOCKS_PER_SEC);
      cout<<"Iteration " << itr << " Completed (Time: " << diff << " sec)"<<endl;
      dft(dataset->objF ,dataset->objCrop, DFT_INVERSE | DFT_SCALE);
   }
     //showImgObject((dataset->objCrop), "Object");
     //showImgFourier((dataset->objF),"Object Spectrum");
     //showImgObject(fftShift(dataset->pupil),"Pupil");
     t4=clock();
     float diff (((float)t4-(float)t3) / CLOCKS_PER_SEC);
     cout<<"FP Processing Completed (Time: " << diff << " sec)"<<endl;
     
     //showComplexImg(dataset->objF, SHOW_COMPLEX_MAG, "Object Spectrum");
     showComplexImg(dataset->objCrop, SHOW_COMPLEX_COMPONENTS, "Object");
     fftShift(dataset->pupil,dataset->pupil);
     showComplexImg(dataset->pupil, SHOW_COMPLEX_COMPONENTS, "Pupil");
}

int main(int argc, char** argv )
{
   // Parameters from the .m file
   uint16_t Np = 90;
   FPM_Dataset mDataset;
   mDataset.datasetRoot = "/home/zfphil/Dropbox/Repository/Datasets/FP_mono_nofilter/";
   mDataset.ledCount = 508;
   mDataset.pixelSize = 6.5;
   mDataset.objectiveMag = 4*2;
   mDataset.objectiveNA = 0.2;
   mDataset.maxIlluminationNA = 0.7604;
   mDataset.color = false;
   mDataset.centerLED = 249;
   mDataset.lambda = 0.45; // um
   mDataset.ps_eff = mDataset.pixelSize / (float) mDataset.objectiveMag;
   mDataset.du= (1/mDataset.ps_eff)/(float) Np;
   std::cout << "Dataset Root: " << mDataset.datasetRoot << std::endl;
   
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
   mDataset.delta2 = 10;
   
   mDataset.itrCount = 10;
   
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
   
   /*
   Mat planes[] = {Mat::zeros(mDataset.Np,mDataset.Np, CV_64F), Mat::zeros(mDataset.Np,mDataset.Np, CV_64F)};
   planes[0] = Mat::zeros(mDataset.Np,mDataset.Np, CV_64F);
   planes[1] = Mat::zeros(mDataset.Np,mDataset.Np, CV_64F);
   
   Mat ft; Mat realObj; Mat realObj2;
   

   float offset = 11;
   float ctrX = 45;
   float ctrY = 45;
   planes[0].at<double>(ctrY,ctrX+offset) = (double)1.00;
   planes[0].at<double>(ctrY,ctrX-offset) = (double)1.00;
   
   //planes[1].at<double>(ctrY+offset,ctrX) = (double) 5.00;
   //planes[1].at<double>(ctrY-offset,ctrX) = (double)10.00;
   
   merge(planes,2,realObj);
   
   showImgObject(realObj, "realObj");
   realObj2 = ifftShift(realObj);
   //fft2(realObj,ft);
   cv::dft(realObj2, ft, DFT_COMPLEX_OUTPUT);
   showImgObject(ft, "FT");
   //ifft2(ft,realObj2);
   

   cv::dft(ft,realObj2, DFT_INVERSE | DFT_COMPLEX_OUTPUT | DFT_SCALE); // Real-space of object
   realObj = fftShift(realObj2);
   showImgObject(realObj,"Result");
   */
   
   loadDataset(&mDataset);
   
   runFPM(&mDataset);
   
   //saveFullDataset(&mDataset, "tmpDataset/");
   /*
   showImg(mDataset.imageStack.at(249).Image);
   imwrite("tmp.tiff",mDataset.imageStack.at(249).Image);
  // fpmBackgroundSubtraction(&mDataset);
  */
   
}