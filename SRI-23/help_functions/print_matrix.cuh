template <typename T>
__device__
void printMatrix(T* matrix, uint32_t rows, uint32_t cols) {
  for (unsigned i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%lf  ", matrix[j*rows+i]); 
    }
    printf("\n");
  }
}