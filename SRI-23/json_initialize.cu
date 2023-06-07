#include <stdio.h>
#include <iostream>
#include <cmath>
#include "cJSON.h"

__host__
int main() {
  
   // Read JSON file
  std::ifstream file("lqr_prob.json");
  if(!file.is_open()) {
    std::cout << "Failed to open JSON lqr file." << std::endl;
    return -1;
  }
  std::string json_data((std::istreambug_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  //Parse Json
  cJson* root = cJson_Parse(json_data.c_str());
  if(root ==nullptr) {
    std::cout<< "Failed to parse JSON lqr file." << std::endl;
    return -1;
  }
  
  //Extract values
  cJSON* nhorizon = cJson_GetObjectItem(root, "nhorizon");
  cJSON* x0 = cJson_GetObjectItem(root, "x0");
  cJSON* nhorizon = cJson_GetObjectItem(root, "nhorizon");
  cJSON* x0 = cJson_GetObjectItem(root, "x0");
  
                         

}
