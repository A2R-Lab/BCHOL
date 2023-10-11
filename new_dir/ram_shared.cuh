// move shared to RAM

// Q_R
for (unsigned i = thread_id; i < (states_sq + inputs_sq) * nhorizon; i += block_dim)
{
    d_Q_R[i] = s_Q_R[i];
}
block.sync();

// q_r
for (unsigned i = thread_id; i < (nstates + ninputs) * nhorizon; i += block_dim)
{
    d_q_r[i] = s_q_r[i];
}
block.sync();

// A_B
for (unsigned i = thread_id; i < (states_sq + inp_states) * nhorizon; i += block_dim)
{
    d_A_B[i] = s_A_B[i];
}
block.sync();

// d
for (unsigned i = thread_id; i < nhorizon * nstates; i += block_dim)
{
    d_d[i] = s_d[i];
}
block.sync();

// F_lambda,F_state,F_input

for (unsigned i = thread_id; i < (states_sq)*nhorizon * depth; i += block_dim)
{
    d_F_lambda[i] = s_F_lambda[i];
    d_F_state[i] = s_F_state[i];
}
for (unsigned i = thread_id; i < (inp_states)*nhorizon * depth; i += block_dim)
{
    d_F_input[i] = s_F_input[i];
}

// move RAM to shared

template <typename T>
__device__ void ramToShar(T *ram, T *shar, uint32_t size)
{
    // block/threads init
    const cgrps::thread_block block = cgrps::this_thread_block();
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;

    for ( unsigned i = thread_id; i < size; i+=block_dim) {
        shar[i] = ram[i];
    }
}

//you have copy function for this!