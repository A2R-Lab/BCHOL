if(DEBUG) {
    printf("PURPOSE");
    for(unsigned i = 0; i < nhorizon; i++) { 
        printf("\nd #%d: \n", i);
        printMatrix(s_d+i*nstates,1,nstates);      

        printf("\nq #%d: \n", i);
        printMatrix(s_q_r+(i*(ninputs+nstates)),1,nstates);

        printf("\nr #%d: \n", i);
        printMatrix(s_q_r+(i*(ninputs+nstates)+nstates),1,ninputs);

        printf("\nA #%d: \n", i);
        printMatrix(s_A_B+(i*dyn_step),nstates,nstates);

        printf("\nB #%d: \n", i);
        printMatrix(s_A_B+(i*dyn_step+states_sq), ninputs, nstates);

        printf("\nQ #%d: \n", i);
        printMatrix(s_Q_R+(i*cost_step),nstates,nstates);

        printf("\nR #%d: \n", i);
        printMatrix(s_Q_R+(i*cost_step+states_sq), ninputs, ninputs);

    }
    for(uint32_t ind = 0; ind < nhorizon * depth ;  ind++) {
        if(ind%nhorizon==0){ 
        printf("\nLEVEL %d\n", ind/nhorizon);
        }
        printf("\nF_lambda #%d: \n", ind);
        printMatrix(s_F_lambda+(ind*states_sq),nstates,nstates);

        printf("\nF_state #%d: \n", ind);
        printMatrix(s_F_state+(ind*states_sq),nstates,nstates);

        printf("\nF_input #%d: \n", ind);
        printMatrix(s_F_input+ind*inp_states, ninputs, nstates);

    }
}