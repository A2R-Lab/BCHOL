template <typename T>
__device__ void ramToshare( uint32_t nhorizon,
                            uint32_t ninputs,
                            uint32_t nstates,
                            T *s_Q_R, T *d_Q_R,
                            T *s_q_r, T *d_q_r,
                            T *s_A_B, T *d_A_B,
                            T *s_d,   T *d_d,
                            T *s_F_lambda, T *d_F_lambda,
                            T *s_F_state,  T *d_F_state,
                            T *s_F_input,  T *d_F_input)
{
    
    glass::copy<float>(cost_step*nhorizon, d_Q_R, s_Q_R);
    glass::copy<float>((nstates+ninputs)*nhorizon, d_q_r,s_q_R);
    glass::copy<float>((states_sq+inp_states)*nhorizon, d_A_B, s_A_B);
    glass::copy<float>(nstates*nhorizon,d_d,s_d);
    //not sure if we need F
    glass::copy<float>(states_sq*nhorizon, d_F_lambda,s_F_lambda);
    glass::copy<float>(states_sq*nhorizon,d_F_state,s_F_state);
    glass::copy<float>(inp_states*nhorizon,d_F_input,s_F_input);
}

template <typename T>
__device__ void sharedToRam(T *s_Q_R, T *d_Q_R,
                            T *s_q_r, T *d_q_r,
                            T *s_A_B, T *d_A_B,
                            T *s_d,   T *d_d,
                            T *s_F_lambda, T *d_F_lambda,
                            T *s_F_state,  T *d_F_state,
                            T *s_F_input,  T *d_F_input)
{
    glass::copy<float>(cost_step*nhorizon, s_Q_R, d_Q_R);
    glass::copy<float>((nstates+ninputs)*nhorizon, s_q_r,d_q_r);
    glass::copy<float>((states_sq+inp_states)*nhorizon, s_A_B, d_A_B);
    glass::copy<float>(nstates*nhorizon,s_d,d_d);
    glass::copy<float>(states_sq*nhorizon, s_F_lambda,d_F_lambda);
    glass::copy<float>(states_sq*nhorizon,s_F_state,d_F_state);
    glass::copy<float>(inp_states*nhorizon,s_F_input,d_F_input);
}