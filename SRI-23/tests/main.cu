/*
A file to run the test kernel and run all the unit tests

*/

// An experiment file to remind myself how I/O works in C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include "test_chol.cuh"
#include "test_chol_InPlace.cuh"

using namespace std;
int main()
{
    printf("HI\n");
    test_chol_InPlace();
    return 0;
}