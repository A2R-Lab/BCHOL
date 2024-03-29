#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>

/** @brief builds the binary tree into an array form
 * @param nhorizon number of timesteps
 * @param levels the array that will hold binary tree structure
 */
__device__ void initializeBSTLevels(int nhorizon, int *levels)
{
  int depth = log2f(nhorizon);

  for (int i = 0; i < nhorizon / 2; i++)
  {
    levels[2 * i] = 0;
    levels[2 * i + 1] = -1;
  }

  bool toggle = false;
  int previous_index = -1;
  for (int i = 1; i < depth; i++)
  {
    for (int j = 0; j < nhorizon; j++)
    {
      if (levels[j] != i - 1)
      {
        continue;
      }
      if (toggle)
      {
        levels[(previous_index + j) / 2] = i;
        toggle = false;
      }
      else
      {
        previous_index = j;
        toggle = true;
      }
    }
  }
}

/** @brief Gets all numbers at a specific level of the binary tree
 * @param nhorizon Number of timesteps
 * @param levels The array that holds the binary tree structure
 * @param targetLevel The level for which to retrieve the numbers
 * @param result An array to store the values at the specified level
 * @return count Number of timesteps at @p targetLevel
 */
__device__ int getValuesAtLevel(int nhorizon, const int *levels, int targetLevel, int *result)
{
  // Initialize the result array to ensure it's "cleaned" before each call
  for (int i = 0; i < nhorizon; i++)
  {
    result[i] = -1; // Use -1 to idicate no results
  }

  int count = 0;

  for (int i = 0; i < nhorizon; i++)
  {
    if (levels[i] == targetLevel)
    {
      result[count++] = i;
    }
  }
  return count;
}
