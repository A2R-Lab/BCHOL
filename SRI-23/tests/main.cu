/*
A file to run the test kernel and run all the unit tests

*/

// An experiment file to remind myself how I/O works in C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include "csv.cuh"
#include "test_kernel.cuh"
#include "check_equality_test.cuh"
#include "test_chol_InPlace.cuh"

using namespace std;
int main()
{
    test_chol_InPlace();
    return 0;
}