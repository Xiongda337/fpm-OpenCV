


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
