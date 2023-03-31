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


import numpy as np
import sys

class LQRData:
    def __init__(self, nstates, ninputs):
        self.nstates = nstates
        self.ninputs = ninputs
        self.Q = np.zeros(nstates, dtype=np.float64)
        self.R = np.zeros(ninputs, dtype=np.float64)
        self.q = np.zeros(nstates, dtype=np.float64)
        self.r = np.zeros(ninputs, dtype=np.float64)
        self.c = np.zeros(1, dtype=np.float64)
        self.A = np.zeros((nstates, nstates), dtype=np.float64)
        self.B = np.zeros((nstates, ninputs), dtype=np.float64)
        self.d = np.zeros(nstates, dtype=np.float64)

    def ndlqr_InitializeLQRData(self, Q, R, q, r, c, A, B, d):
        nstates = self.nstates
        ninputs = self.ninputs
        np.copyto(self.Q, Q)
        np.copyto(self.R, R)
        np.copyto(self.q, q)
        np.copyto(self.r, r)
        np.copyto(self.c, c)
        np.copyto(self.A, A)
        np.copyto(self.B, B)
        np.copyto(self.d, d)

    def __str__(self):
        return (f"LQR Data with n={self.nstates}, m={self.ninputs}:\n"
                f"Q = {self.Q}\n"
                f"R = {self.R}\n"
                f"q = {self.q}\n"
                f"r = {self.r}\n"
                f"c = {self.c}\n"
                f"A = {self.A}\n"
                f"B = {self.B}\n"
                f"d = {self.d}"
               )

def ndlqr_CopyLQRData(dest, src):
  if dest.nstates != src.nstates or dest.ninputs != src.ninputs:
    print(f"Can't copy LQRData of different sizes: ({dest.nstates},{dest.ninputs}) and ({src.nstates},{src.ninputs}).", file=sys.stderr)
    return -1
  dest.ndlqr_InitializeLQRData(src.Q, src.R, src.q, src.r, src.c, src.A, src.B, src.d)
  return 0
