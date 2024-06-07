#include <stdio.h>
#include <iostream>
#include <numeric>
#include <cmath>
#include "solve.cuh"
#include "./help_functions/csv.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include "gpu_assert.cuh"

__host__ int main()
{
  printf("Run Test\n");
  // Declaration of LQR problem
  uint32_t knot_points = 8;
  uint32_t state_size = 6;
  uint32_t control_size = 3;
  uint32_t depth = log2(knot_points);

  // calculating the constants
  const uint32_t states_sq = state_size * state_size;
  const uint32_t controls_sq = control_size * control_size;
  const uint32_t states_p_controls = state_size * control_size;
  const uint32_t states_s_controls = state_size + control_size;
  const uint32_t fstates_size = states_sq * knot_points * depth;
  const uint32_t fcontrol_size = states_p_controls * knot_points * depth;

  const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq + controls_sq) * knot_points - controls_sq) * sizeof(float));
  const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq + states_p_controls) * (knot_points - 1) * sizeof(float));
  const uint32_t KKT_g_SIZE_BYTES = static_cast<uint32_t>(((state_size + control_size) * knot_points - control_size) * sizeof(float));
  const uint32_t KKT_c_SIZE_BYTES = static_cast<uint32_t>((state_size * knot_points) * sizeof(float));
  const uint32_t KKT_FSTATES_SIZE_BYTES = static_cast<uint32_t>(fstates_size * sizeof(float));
  const uint32_t KKT_FCONTROL_SIZE_BYTES = static_cast<uint32_t>(fcontrol_size * sizeof(float));

  // const uint32_t DZ_SIZE_BYTES = static_cast<uint32_t>((states_s_controls * knot_points - control_size) * sizeof(float));

  float Q_R[((states_sq + controls_sq) * knot_points - controls_sq)]={1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q0
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R0
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q1
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R1
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q2
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R2
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q3
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R3
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q4
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R4
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q5
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R5
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q6
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R6
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   10.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q7
                                                                   0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 10.0}; 
  float q_r[((state_size + control_size) * knot_points - control_size)]={
      -2.0, -1.2, -0.4, 0.4, 1.2, 2.0,                                                             // q0
      -1.0, 0.0, 1.0,                                                                              // r0
      -4.0, -2.4, -0.8, 0.8, 2.4, 4.0,                                                             // q1
      -2.0, 0.0, 2.0,                                                                              // r1
      -6.0, -3.5999999999999996, -1.2000000000000002, 1.2000000000000002, 3.5999999999999996, 6.0, // q2
      -3.0, 0.0, 3.0,                                                                              // r2
      -8.0, -4.8, -1.6, 1.6, 4.8, 8.0,                                                             // q3
      -4.0, 0.0, 4.0,                                                                              // r3
      -10.0, -6.0, -2.0, 2.0, 6.0, 10.0,                                                           // q4
      -5.0, 0.0, 5.0,                                                                              // r4
      -12.0, -7.199999999999999, -2.4000000000000004, 2.4000000000000004, 7.199999999999999, 12.0, // q5
      -6.0, 0.0, 6.0,                                                                              // r5
      -14.0, -8.4, -2.8000000000000003, 2.8000000000000003, 8.4, 14.0,                             // q6
      -7.0, 0.0, 7.0,                                                                              // r6
      -160.0, -96.0, -32.0, 32.0, 96.0, 160.0                                   // q7
  };       
  float A_B[(states_sq + states_p_controls) * (knot_points - 1)]={
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A0 or column of A0^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A0
      0.005000000000000001, 0.0, 0.0, // row of B0 or column of B0^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B0
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A1 or column of A1^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A1
      0.005000000000000001, 0.0, 0.0, // row of B1 or column of B1^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B1
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A2 or column of A2^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A2
      0.005000000000000001, 0.0, 0.0, // row of B2 or column of B2^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B2
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A3 or column of A3^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A3
      0.005000000000000001, 0.0, 0.0, // row of B3 or column of B3^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B3
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A4 or column of A4^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A4
      0.005000000000000001, 0.0, 0.0, // row of B4 or column of B4^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B4
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A5 or column of A5^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A5
      0.005000000000000001, 0.0, 0.0, // row of B5 or column of B5^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B5
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A6 or column of A6^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A6
      0.005000000000000001, 0.0, 0.0, // row of B6 or column of B6^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1            // end of B6
  };
  float d[(state_size * knot_points)]={1.0, -1.0, 2.0, -2.0, 3.0, -3.0,   // x0
      1.5, 1.5, 1.5, 1.5, 1.5, 1.5,      // d0
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,      // d1
      4.5, 4.5, 4.5, 4.5, 4.5, 4.5,      // d2
      6.0, 6.0, 6.0, 6.0, 6.0, 6.0,      // d3
      7.5, 7.5, 7.5, 7.5, 7.5, 7.5,      // d4
      9.0, 9.0, 9.0, 9.0, 9.0, 9.0,      // d5
      10.5, 10.5, 10.5, 10.5, 10.5, 10.5};
  uint32_t soln_size = (state_size + state_size + control_size) * knot_points - control_size;
  float soln[soln_size] = { 118.50447730549635,
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
      };
  float my_soln[soln_size];

  //Write CSV
  write_csv("lqr_prob8.csv",knot_points,state_size,control_size,Q_R, q_r, A_B, d, soln);
  // Reading the LQR problem
  read_csv("lqr_prob8.csv", knot_points, state_size, control_size, Q_R, q_r, A_B, d, soln);

  printMatrix(Q_R+6*(states_sq+controls_sq),state_size,state_size);
  // Creating Factorization
  float F_lambda[fstates_size];
  float F_state[fstates_size];
  for (uint32_t n = 0; n < fstates_size; n++)
  {
    F_lambda[n] = 0;
    F_state[n] = 0;
  }

  float F_input[fcontrol_size];
  for (uint32_t n = 0; n < fcontrol_size; n++)
  {
    F_input[n] = 0;
  }

  // Allocate memory on the GPU for x0,Q_R,q_r, A_B, d,

  float *d_Q_R, *d_q_r, *d_A_B, *d_d,
      *d_F_lambda, *d_F_state, *d_F_input;
  gpuErrchk(cudaMalloc((void **)&d_Q_R, KKT_G_DENSE_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_q_r, KKT_g_SIZE_BYTES));
  gpuErrchk((cudaMalloc((void **)&d_A_B, KKT_C_DENSE_SIZE_BYTES)));
  gpuErrchk(cudaMalloc((void **)&d_d, KKT_c_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_F_lambda, KKT_FSTATES_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_F_state, KKT_FSTATES_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_F_input, fcontrol_size * sizeof(float)));
  gpuErrchk(cudaPeekAtLastError());

  // Copy the matrices from the host to the GPU memory
  // cudaMemcpy(d_x0, x0, 6 * sizeof(float), cudaMemcpyHostToDevice);
  gpuErrchk(cudaMemcpy(d_Q_R, Q_R, KKT_G_DENSE_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_q_r, q_r, KKT_g_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_A_B, A_B, KKT_C_DENSE_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_d, d, KKT_c_SIZE_BYTES, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_F_lambda, F_lambda, KKT_FSTATES_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_F_state, F_state, KKT_FSTATES_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_F_input, F_input, KKT_FCONTROL_SIZE_BYTES, cudaMemcpyHostToDevice));

  // Launch CUDA kernel with block and grid dimensions
  // when increasing blocksize to 32 not working
  std::uint32_t blockSize = 32;
  std::uint32_t gridSize = 8;

  uint32_t shared_mem = 5 * 2160 * sizeof(float);

  const void *kernelFunc = reinterpret_cast<const void *>(solve_Kernel<float>);
  void *args[] = {// prepare the kernel arguments
                  &knot_points,
                  &state_size,
                  &control_size,
                  &d_Q_R,
                  &d_q_r,
                  &d_A_B,
                  &d_d,
                  &d_F_lambda,
                  &d_F_state,
                  &d_F_input};
  // Prepare for timing
  cudaEvent_t start, stop;
  float time;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventRecord(start, 0));
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaLaunchCooperativeKernel(kernelFunc, gridSize, blockSize, args, shared_mem));
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());

  // Copy back to the host
  cudaMemcpy(q_r, d_q_r, KKT_g_SIZE_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(d, d_d, KKT_c_SIZE_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(Q_R, d_Q_R, KKT_G_DENSE_SIZE_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(A_B, d_A_B, KKT_C_DENSE_SIZE_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(F_lambda, d_F_lambda, KKT_FSTATES_SIZE_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(F_state, d_F_state, KKT_FSTATES_SIZE_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(F_input, d_F_input, KKT_FCONTROL_SIZE_BYTES, cudaMemcpyDeviceToHost);

  gpuErrchk(cudaEventRecord(stop, 0));
  gpuErrchk(cudaEventSynchronize(stop));
  gpuErrchk(cudaEventElapsedTime(&time, start, stop));
  printf("\nSolve Time:  %3.1f ms \n", time);

  for (uint32_t timestep = 0; timestep < knot_points; ++timestep)
  {
    for (uint32_t i = 0; i < state_size; ++i)
    {
      my_soln[timestep * (state_size + state_size + control_size) + i] = d[timestep * state_size + i];
    }
    for (uint32_t i = 0; i < states_s_controls; ++i)
    {
      my_soln[timestep * (state_size + state_size + control_size) + state_size + i] = q_r[timestep * (states_s_controls) + i];
    }
  }

  if (checkEquality(my_soln, soln, soln_size))
  {
    printf("PASSED!\n");
  }
  else
  {
    printf("Not Passed");
    printf("my_soln\n");
    printMatrix(my_soln, (state_size + state_size + control_size) * 2, 1);
    printf("Soln\n");
    printMatrix(soln, (state_size + state_size+ control_size) * 2, 1);
  }

  if (!true)
  {
    printf("CHECK FINAL RESULTS on host\n");

    for (unsigned i = 0; i < knot_points; i++)
    {

      printMatrix(d + i * state_size, state_size, 1);
      printMatrix(q_r + (i * (state_size + state_size)), state_size, 1);
      printMatrix(q_r + (i * (control_size + state_size) + state_size), control_size, 1);
    }
  }

  // Free allocated GPU memory
  gpuErrchk(cudaFree(d_Q_R));
  gpuErrchk(cudaFree(d_q_r));
  gpuErrchk(cudaFree(d_A_B));
  gpuErrchk(cudaFree(d_d));
  gpuErrchk(cudaFree(d_F_lambda));
  gpuErrchk(cudaFree(d_F_state));
  gpuErrchk(cudaFree(d_F_input));
}