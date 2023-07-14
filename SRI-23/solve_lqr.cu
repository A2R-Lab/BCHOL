#include <stdio.h>
#include <iostream>
#include <cmath>
#include "solve.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
//#include "blockassert.cuh" //need to write!


__host__
int main() {
  printf("Run Test\n");
  //Info of LQR problem
  const int nhorizon = 8;
  const int nstates = 6;
  const int ninputs = 3; 
  const int depth = log2f(nhorizon); // calculate depth of tree from nhorizon
  const int states_sq = nstates*nstates;
  const int inputs_sq = ninputs*ninputs;
  const int inp_states = ninputs*nstates;
  const int cost_step = states_sq+inputs_sq;
  const int dyn_step = states_sq+inp_states;
  const int Q_R_size = cost_step*nhorizon;
  const int q_r_size = (nstates+ninputs)*nhorizon;
  const int A_B_size = dyn_step*nhorizon;
  const int d_size = nstates*nhorizon;
  const int F_lambda_size = states_sq*nhorizon*depth;
  const int F_state_size = states_sq*nhorizon*depth;
  const int F_input_size = inp_states*nhorizon*depth;

  double Q_R[Q_R_size] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q0
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           0.01, 0.00, 0.00, //R0
                           0.00, 0.01, 0.00,
                           0.00, 0.00, 0.01,
                           1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q1
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           0.01, 0.00, 0.00, //R1
                           0.00, 0.01, 0.00,
                           0.00, 0.00, 0.01,
                           1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q2
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           0.01, 0.00, 0.00, //R2
                           0.00, 0.01, 0.00,
                           0.00, 0.00, 0.01,
                           1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q3
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           0.01, 0.00, 0.00, //R3
                           0.00, 0.01, 0.00,
                           0.00, 0.00, 0.01,
                           1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q4
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           0.01, 0.00, 0.00, //R4
                           0.00, 0.01, 0.00,
                           0.00, 0.00, 0.01,
                           1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q5
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           0.01, 0.00, 0.00, //R5
                           0.00, 0.01, 0.00,
                           0.00, 0.00, 0.01,
                           1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q6
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           0.01, 0.00, 0.00, //R6
                           0.00, 0.01, 0.00,
                           0.00, 0.00, 0.01,
                           10.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q7
                           0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
                           0.00, 0.00, 0.00, //R7
                           0.00, 0.00, 0.00,
                           0.00, 0.00, 0.00
                        }; //Q_R diagonal matrices

   double q_r[q_r_size] = {-2.0, -1.2, -0.4, 0.4, 1.2, 2.0, //q0
                     -1.0,  0.0,  1.0, //r0
                     -4.0, -2.4, -0.8, 0.8, 2.4, 4.0, //q1
                     -2.0,  0.0,  2.0, //r1
                     -6.0, -3.5999999999999996, -1.2000000000000002, 1.2000000000000002, 3.5999999999999996, 6.0, //q2   
                     -3.0,  0.0, 3.0, //r2
                     -8.0, -4.8, -1.6, 1.6, 4.8, 8.0, //q3
                     -4.0,  0.0,  4.0, //r3
                     -10.0,-6.0, -2.0, 2.0, 6.0, 10.0, //q4
                     -5.0, 0.0, 5.0, //r4
                     -12.0, -7.199999999999999, -2.4000000000000004, 2.4000000000000004, 7.199999999999999, 12.0, //q5
                     -6.0,  0.0, 6.0, //r5
                     -14.0, -8.4, -2.8000000000000003, 2.8000000000000003, 8.4, 14.0, //q6
                     -7.0,  0.0, 7.0, //r6
                     -160.0, -96.0, -32.0, 32.0, 96.0, 160.0 //q7
                     -0.0, 0.0, 0.0 //r7 - all zeros
                     }; //vectors q_r            
                    
   double A_B[A_B_size] = {   1.0, 0.0, 0.0, 0.1, 0.0, 0.0, //row of A0 or column of A0^T
                              0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //end of A0
                              0.005000000000000001, 0.0, 0.0, //row of B0 or column of B0^T
                              0.0, 0.005000000000000001, 0.0,
                              0.0, 0.0, 0.005000000000000001, 
                              0.1, 0.0, 0.0,
                              0.0, 0.1, 0.0,
                              0.0, 0.0, 0.1, //end of B0
                              1.0, 0.0, 0.0,	0.1, 0.0, 0.0, //row of A1 or column of A1^T
                              0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //end of A1
                              0.005000000000000001, 0.0,	0.0, //row of B1 or column of B1^T
                              0.0, 0.005000000000000001, 0.0,
                              0.0, 0.0, 0.005000000000000001,
                              0.1, 0.0, 0.0,
                              0.0, 0.1, 0.0,
                              0.0, 0.0, 0.1, //end of B1
                              1.0, 0.0, 0.0, 0.1, 0.0, 0.0, //row of A2 or column of A2^T
                              0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //end of A2
                              0.005000000000000001, 0.0, 0.0, //row of B2 or column of B2^T
                              0.0, 0.005000000000000001, 0.0,
                              0.0, 0.0, 0.005000000000000001, 
                              0.1, 0.0, 0.0,
                              0.0, 0.1, 0.0,
                              0.0, 0.0, 0.1, //end of B2
                              1.0, 0.0, 0.0,	0.1, 0.0, 0.0, //row of A3 or column of A3^T
                              0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //end of A3
                              0.005000000000000001, 0.0,	0.0, //row of B3 or column of B3^T
                              0.0, 0.005000000000000001, 0.0,
                              0.0, 0.0, 0.005000000000000001,
                              0.1, 0.0, 0.0,
                              0.0, 0.1, 0.0,
                              0.0, 0.0, 0.1, //end of B3
                              1.0, 0.0, 0.0, 0.1, 0.0, 0.0, //row of A4 or column of A4^T
                              0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //end of A4
                              0.005000000000000001, 0.0, 0.0, //row of B4 or column of B4^T
                              0.0, 0.005000000000000001, 0.0,
                              0.0, 0.0, 0.005000000000000001, 
                              0.1, 0.0, 0.0,
                              0.0, 0.1, 0.0,
                              0.0, 0.0, 0.1, //end of B4
                              1.0, 0.0, 0.0,	0.1, 0.0, 0.0, //row of A5 or column of A5^T
                              0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //end of A5
                              0.005000000000000001, 0.0,	0.0, //row of B5 or column of B5^T
                              0.0, 0.005000000000000001, 0.0,
                              0.0, 0.0, 0.005000000000000001,
                              0.1, 0.0, 0.0,
                              0.0, 0.1, 0.0,
                              0.0, 0.0, 0.1, //end of B5
                              1.0, 0.0, 0.0,	0.1, 0.0, 0.0, //row of A6 or column of A6^T
                              0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //end of A6
                              0.005000000000000001, 0.0,	0.0, //row of B6 or column of B6^T
                              0.0, 0.005000000000000001, 0.0,
                              0.0, 0.0, 0.005000000000000001,
                              0.1, 0.0, 0.0,
                              0.0, 0.1, 0.0,
                              0.0, 0.0, 0.1, //end of B6
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A7or column of A7^T
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //end of A7
                              0.0, 0.0, 0.0, //row of B7or column of B7^T
                              0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0 //end of B7
                           };
                
   double d[48] = {  1.0, -1.0, 2.0, -2.0, 3.0, -3.0, //x0
                     1.5, 1.5, 1.5, 1.5, 1.5, 1.5, //d0
                     3.0, 3.0, 3.0, 3.0, 3.0, 3.0, //d1
                     4.5, 4.5, 4.5,4.5, 4.5, 4.5, //d2
                     6.0, 6.0, 6.0,6.0, 6.0, 6.0,  //d3
                     7.5, 7.5, 7.5,7.5, 7.5, 7.5,  //d4
                     9.0, 9.0, 9.0,9.0, 9.0, 9.0, // d5
                     10.5,10.5,10.5,10.5,10.5, 10.5 //d6
                  };

   //Allocate memory on the GPU
   double* d_Q_R;
   cudaMalloc((void**)&d_Q_R, Q_R_size*sizeof(double));
   double* d_q_r;
   cudaMalloc((void**)&d_q_r, q_r_size*sizeof(double));
   double* d_A_B;
   cudaMalloc((void**)&d_A_B, A_B_size*sizeof(double));
   double* d_d;
   cudaMalloc((void**)&d_d, d_size*sizeof(double));
   double* d_F_lambda;
   cudaMalloc((void**)&d_F_lambda, F_lambda_size*sizeof(double));
   double* d_F_state;
   cudaMalloc((void**)&d_F_state, F_state_size*sizeof(double));
   double* d_F_input;
   cudaMalloc((void**)&d_F_input, F_input_size*sizeof(double));
   
   printf("Allocated memory\n");
   
   //Copy the matrices from the host to the GPU memory
   cudaMemcpy(d_Q_R, Q_R, Q_R_size*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(d_q_r, q_r, q_r_size*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(d_A_B, A_B, A_B_size*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(d_d, d, d_size*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemset(d_F_lambda, 0, F_lambda_size*sizeof(double));
   cudaMemset(d_F_state, 0, F_state_size*sizeof(double));
   cudaMemset(d_F_input, 0, F_input_size*sizeof(double));

   uint32_t shared_mem = 0;
   shared_mem += Q_R_size;
   shared_mem += q_r_size;
   shared_mem += A_B_size;
   shared_mem += d_size;
   shared_mem += F_lambda_size;
   shared_mem += F_state_size;
   shared_mem += F_input_size;
   shared_mem *= sizeof(double);
   shared_mem += nhorizon * sizeof(int);

   //Launch CUDA kernel with block and grid dimensions
   int info[] = {nhorizon,ninputs,nstates,depth};
   int* d_info;
   cudaMalloc((void**)&d_info, 4*sizeof(int));
   cudaMemcpy(d_info, info, 4*sizeof(int), cudaMemcpyHostToDevice);

   std::uint32_t blockSize = 1;
   std::uint32_t gridSize = 1;
   const void* kernelFunc = reinterpret_cast<const void*>(solve_Kernel<double>);
   void* args[] = {
      &d_info,
      &d_Q_R,
      &d_q_r,
      &d_A_B,
      &d_d,
      &d_F_lambda,
      &d_F_state,
      &d_F_input
   };
   cudaLaunchCooperativeKernel(kernelFunc, gridSize, blockSize, args, shared_mem);
   cudaDeviceSynchronize();
   printf("BYE!");
   
   //here can either launch one Kernel and call all functions within it and use blocks (cprgs)
   //or can potentially launch a kernel per each big function (solve_leaf etc)
   
   
   //Copy back to the host
   cudaMemcpy(q_r,d_q_r, 72*sizeof(double),cudaMemcpyDeviceToHost);
   cudaMemcpy(d,d_d, 48*sizeof(double),cudaMemcpyDeviceToHost);
   // cudaMemcpy(Q_R,d_Q_R, 360*sizeof(double),cudaMemcpyDeviceToHost);
   // cudaMemcpy(A_B,d_A_B, 432*sizeof(double),cudaMemcpyDeviceToHost);

   //Free allocated GPU memory
   cudaFree(d_info);
   cudaFree(d_Q_R);
   cudaFree(d_q_r);
   cudaFree(d_A_B);
   cudaFree(d_d);
   cudaFree(d_F_lambda);
   cudaFree(d_F_state);
   cudaFree(d_F_input);
}
