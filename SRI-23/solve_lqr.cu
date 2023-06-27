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
//Info about LQR problem

  const uint32_t nhorizon = 8;
  const uint32_t nstates = 6;
  const uint32_t ninputs = 3; 
  const uint32_t depth = 2; 
  
  //float x0[6] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0}; //instead put it as d0
  float Q_R[360] = {  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q0
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                     0.01, 0.00, 0.00, //R0
                     0.00, 0.01, 0.00,
                     0.00, 0.00, 0.01,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q1
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                     0.01, 0.00, 0.00, //R1
                     0.00, 0.01, 0.00,
                     0.00, 0.00, 0.01,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q2
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                     0.01, 0.00, 0.00, //R2
                     0.00, 0.01, 0.00,
                     0.00, 0.00, 0.01,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q3
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                     0.01, 0.00, 0.00, //R3
                     0.00, 0.01, 0.00,
                     0.00, 0.00, 0.01,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q4
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                     0.01, 0.00, 0.00, //R4
                     0.00, 0.01, 0.00,
                     0.00, 0.00, 0.01,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q5
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                     0.01, 0.00, 0.00, //R5
                     0.00, 0.01, 0.00,
                     0.00, 0.00, 0.01,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q6
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                     0.01, 0.00, 0.00, //R6
                     0.00, 0.01, 0.00,
                     0.00, 0.00, 0.01,
                     10.0, 0.0, 0.0, 0.0, 0.0, 0.0, //Q7
                     0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 10.0,
                     0.00, 0.00, 0.00, //R7
                     0.00, 0.00, 0.00,
                     0.00, 0.00, 0.00
                     }; //Q_R diagonal matrices - doesn't matter row/column order

  float q_r[72] = { -2.0, -1.2, -0.4, 0.4, 1.2, 2.0, //q0
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
                    -0.0, 0.0, 0.0 //r7 -doesn't exist
                    }; //vectors q_r
                                     
                    
                    
 //c = 1 (?)
  float A_B[432]= {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A0 or column of A0^T
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.1, 0.0, 0.0, 1.0, //end of A0
                  0.005000000000000001, 0.0, 0.0, 0.1, 0.0, 0.0, //row of B0 or column of B0^T
                  0.0, 0.005000000000000001, 0.0, 0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005000000000000001, 0.0, 0.0, 0.1, //end of B0
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A1 or column of A1^T
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.1, 0.0, 0.0, 1.0, //end of A1
                  0.005000000000000001, 0.0, 0.0, 0.1, 0.0, 0.0, //row of B1 or column of B1^T
                  0.0, 0.005000000000000001, 0.0, 0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005000000000000001, 0.0, 0.0, 0.1, //end of B1
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A2 or column of A2^T
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.1, 0.0, 0.0, 1.0, //end of A2
                  0.005000000000000001, 0.0, 0.0, 0.1, 0.0, 0.0, //row of B2 or column of B2^T
                  0.0, 0.005000000000000001, 0.0, 0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005000000000000001, 0.0, 0.0, 0.1, //end of B2
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A3 or column of A3^T
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.1, 0.0, 0.0, 1.0, //end of A3
                  0.005000000000000001, 0.0, 0.0, 0.1, 0.0, 0.0, //row of B3 or column of B3^T
                  0.0, 0.005000000000000001, 0.0, 0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005000000000000001, 0.0, 0.0, 0.1, //end of B3
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A4 or column of A4^T
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.1, 0.0, 0.0, 1.0, //end of A4
                  0.005000000000000001, 0.0, 0.0, 0.1, 0.0, 0.0, //row of B4 or column of B4^T
                  0.0, 0.005000000000000001, 0.0, 0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005000000000000001, 0.0, 0.0, 0.1, //end of B4
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A5 or column of A5^T
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.1, 0.0, 0.0, 1.0, //end of A5
                  0.005000000000000001, 0.0, 0.0, 0.1, 0.0, 0.0, //row of B5 or column of B5^T
                  0.0, 0.005000000000000001, 0.0, 0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005000000000000001, 0.0, 0.0, 0.1, //end of B5
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A6 or column of A6^T
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.1, 0.0, 0.0, 1.0, //end of A6
                  0.005000000000000001, 0.0, 0.0, 0.1, 0.0, 0.0, //row of B6 or column of B6T
                  0.0, 0.005000000000000001, 0.0, 0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005000000000000001, 0.0, 0.0, 0.1, //end of B6
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of A7or column of A7^T
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //end of A7
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //row of B7or column of B7^T
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //end of B7
                };
    float d[48] = { 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, //x0
                 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, //d0
                 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, //d1
                 4.5, 4.5, 4.5,4.5, 4.5, 4.5, //d2
                 6.0, 6.0, 6.0,6.0, 6.0, 6.0,  //d3
                 7.5, 7.5, 7.5,7.5, 7.5, 7.5,  //d4
                 9.0, 9.0, 9.0,9.0, 9.0, 9.0, // d5
                 10.5,10.5,10.5,10.5,10.5, 10.5 //d6
               };
  
   float soln[117] = { 118.50447730549635,
      172.84569760649603,
      273.5288554044134,
      29.888509228935593,
      30.091008378771242,
      16.84756914227718,
      1.0,
      -1.0,
      2.0,
      -2.0,
      3.0,
      -3.0,
      -155.13285363660776,
      -171.3872349844644,
      -142.51126372056513,
      119.50447730549635,
      175.04569760649602,
      271.92885540441335,
      19.53806149838596,
      8.386438618121641,
      -9.345316398164156,
      1.5243357318169608,
      -0.05693617492232219,
      2.4874436813971745,
      -16.013285363660778,
      -12.638723498446442,
      -15.751126372056515,
      -86.52339783362774,
      -97.50030427497165,
      -88.93739387741547,
      121.98014157367939,
      177.50263378141835,
      270.2414117230162,
      22.5533327046788,
      0.8748987384262467,
      -24.618331198409262,
      2.490390206282744,
      1.1916899538581749,
      3.4676440748044457,
      -21.66562514702355,
      -19.388753925943607,
      -21.644865759798062,
      -67.44470283332527,
      -76.68105472991847,
      -76.27846178978213,
      125.48975136739665,
      179.91094382756017,
      267.9737676482118,
      30.469982714962686,
      -1.3274417183861615,
      -35.77084220343238,
      4.486604177413763,
      3.3694092876142214,
      5.421765189875727,
      -23.910095430356076,
      -22.556859398935455,
      -24.772711938776276,
      -63.299207858196155,
      -73.62340953552,
      -77.94269612427102,
      129.00314718998288,
      181.34153453994597,
      264.1520024583361,
      39.87976342632047,
      -1.7047357734453001,
      -45.413330510489715,
      7.779098595087174,
      6.745606300043076,
      8.554780515376743,
      -24.240016216175693,
      -23.919200352487454,
      -26.56698155120338,
      -55.58577212751382,
      -71.84668167047012,
      -82.73789943565697,
      131.22404859489572,
      180.5959282399029,
      257.5972219429593,
      48.99737478300659,
      -1.8451282449481334,
      -54.60607115358227,
      12.577168112832036,
      11.494452856441978,
      12.984392863078119,
      -22.298593428927074,
      -23.60386851953447,
      -27.340771494769076,
      -23.636241878305015,
      -57.436665054132945,
      -83.84058887192744,
      130.64688048206366,
      176.30147538346088,
      247.0128290798812,
      55.83128016372731,
      -3.071407263759751,
      -63.966582566801314,
      19.2291275605478,
      17.846882679217867,
      18.83111276924157,
      -15.662217616757575,
      -20.347535024947764,
      -26.724830381961823,
      75.77389865590904,
      -5.333981259758644,
      -72.09161999628519,
      125.41775292151588,
      166.854592704243,
      230.98171631063963,
      56.1517224883333,
      -7.809331509236286,
      -74.33992381590345,
      28.541775292151588,
      26.285459270424298,
      26.298171631063962,
      2.4151722488333296,
      -10.380933150923628,
      -23.433992381590347
      }; //soln vector

    /*
   //Fact_Lambda[nstates*nstates*nhorizon*depth]
   float Fact_lambda[36*8*3]; 
   float Fact_state[36*8*3];
   for(std::unit32_t n = 0; n < 864; n++){
      Fact_lambda[n] = 0;
      Fact_state[n] = 0;
   }
  
   //Fact_Input[nstates*ninputs*nhorizon*depth]  
   float Fact_input[18*8*3];
   for(std::unit32_t n = 0; n < 864; n++) {
      Fact_input[n] = 0;
   }

*/

  

      
   //when using for soln_vector need to negate q_r and d_d
   
   
   //Allocate memory on the GPU for x0,Q_R,q_r, A_B, d, 
   /*
   float* d_x0;
   cudaMalloc((void**)&d_x0, 6*sizeof(float));*/

   float* d_Q_R;
   cudaMalloc((void**)&d_Q_R, 360*sizeof(float));

   float* d_q_r;
   cudaMalloc((void**)&d_q_r, 72*sizeof(float));

   float* d_A_B;
   cudaMalloc((void**)&d_A_B, 432*sizeof(float));

   float* d_d;
   cudaMalloc((void**)&d_d, 48*sizeof(float));
   printf("Allocated memory\n");

  /*do we need to allocate memory for F?
   float* d_F_lambda;
   cudaMalloc((void**)&d_F_lambda, 36*8*3*sizeof(float));

   float* d_F_state;
   cudaMalloc((void**)&d_F_state, 36*8*3*sizeof(float));

   float* d_F_input;
   cudaMalloc((void**)&d_F_input, 18*8*3*sizeof(float));
   */
   
   //Copy the matrices from the host to the GPU memory
   //cudaMemcpy(d_x0, x0, 6 * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_Q_R, Q_R, 360 * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_q_r, q_r, 72*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_A_B, A_B, 432*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_d, d, 48*sizeof(float), cudaMemcpyHostToDevice);
/*
   cudaMemcpy(d_F_lambda, F_lambda, 36*8*3*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_F_state, F_state, 36*8*3*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_F_input, F_input, 36*8*3*sizeof(float), cudaMemcpyHostToDevice);
*/

   //Launch CUDA kernel with block and grid dimensions
   std::uint32_t blockSize = 1;
   std::uint32_t gridSize = 1;
   //the arguments allign with solve.cuh - CHECKED
   solve_Kernel<float><<<gridSize, blockSize>>>(nhorizon, ninputs, nstates, d_Q_R, d_q_r, d_A_B, d_d);
   cudaDeviceSynchronize();
    
   
   //here can either launch one Kernel and call all functions within it and use blocks (cprgs)
   //or can potentially launch a kernel per each big function (solve_leaf etc)
   
   
   //Copy back to the host
   cudaMemcpy(q_r,d_q_r, 72*sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(d,d_d, 48*sizeof(float),cudaMemcpyDeviceToHost);

   //Free allocated GPU memory
   cudaFree(d_Q_R);
   cudaFree(d_q_r);
   cudaFree(d_A_B);
   cudaFree(d_d);
}
