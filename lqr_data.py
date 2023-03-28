"""
  @brief Holds the data for a single time step of LQR
 
  Stores the \f$ Q, R, q, r, c \f$ values for the cost function:
  \f[
  \frac{1}{2} x^T Q x + q^T x + \frac{1}{2} u^T R u + r^T r + c
  \f]
 
  and the \f$ A, B, d \f$ values for the dynamics:
  \f[
  x_{k+1} = A x_k + B u_k + d
  \f]
 
  ## Construction and destruction
  A new LQRData object is constructed using ndlqr_NewLQRData(), which must be
  freed with a call to ndlqr_FreeLQRData().
 
  ## Methods
  - ndlqr_NewLQRData()
  - ndlqr_FreeLQRData()
  - ndlqr_InitializeLQRData()
  - ndlqr_CopyLQRData()
  - ndlqr_PrintLQRData()
 
  ## Getters
  The follow methods return a Matrix object wrapping the data from an LQRData object.
  The user should NOT call FreeMatrix() on this data since it is owned by the LQRData
  object.
  - ndlqr_GetA()
  - ndlqr_GetB()
  - ndlqr_Getd()
  - ndlqr_GetQ()
  - ndlqr_GetR()
  - ndlqr_Getq()
  - ndlqr_Getr()
 
 """