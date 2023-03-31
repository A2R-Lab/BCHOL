class LQRProblem (object):
    """
    @brief Describes an LQR problem with affine terms

    Internally, stores a vector of LQRData, one for each knot point, along with
    the initial state and the horizon length.

    ## Construction and desctruction
    The user can initialize and empty problem using ndlqr_NewLQRProblem(),
    which again must be paired with a call to ndlqr_FreeLQRProblem().

    After the problem is initialized, it can be filled in from a vector LQRData using
    ndlqr_InitializeLQRProblem().
    """
    def _init_(self):
        nhorizon
        x0
        lqrdata

    def ndlqr_InitializeLQRProblem(self, lqrproblem, x0, lqrdata):
        """
        @brief Initialize the problem with an initial state and the LQR data

        @param lqrproblem  An initialized LQRProblem
        @param x0          Initial state vector. The data is copied into the problem.
        @param lqrdata     A vector of LQR data. Each element is copied into the problem.
        @return 0 if successful
        """
        if lqrproblem == None:
            return -1
        for k  in range(lqrproblem.nhorizon):
            ndlqr_CopyLQRData(lqrproblem.lqrdata[k], lqrdata[k])
        lqrproblem.x0 = x0
        return 0

    def ndlqr_NewLQRProblem(self, nstates, ninputs, nhorizon):
        """
        @brief Initialize a new LQRProblem data with unitialized data

        Must be paired with a call to ndlqr_FreeLQRProblem().

        @param nstates Length of the state vector
        @param ninputs Number of control inputs
        @param nhorizon Length of the horizon (i.e. number of knot points)
        @return
        """
        if nhorizon <= 0:
            print("ERROR: Horizon must be positive.")
            return None

        lqrdata = # Srishti - "Unsure how to do this"
        for k in range(nhorizon):
            lqrdata[k] = ndlqr_NewLQRData(nstates, ninputs)

        x0 =  # Srishti - "Unsure how to do this"
        lqrproblem =  # Srishti - "Unsure how to do this"
        lqrproblem.nhorizon = nhorizon
        lqrproblem.x0 = x0
        lqrproblem.lqrdata = lqrdata
        return lqrproblem
