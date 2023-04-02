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

    def _init_(self, nstates, ninputs, nhorizon):
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

        for k in range(nhorizon):
            lqrdata[k] = ndlqr_NewLQRData(nstates, ninputs)

        self.nhorizon = nhorizon
        self.x0 = None
        self.lqrdata = lqrdata
        return

    def ndlqr_InitializeLQRProblem(self, x0, lqrdata):
        """
        @brief Initialize the problem with an initial state and the LQR data

        @param lqrproblem  An initialized LQRProblem
        @param x0          Initial state vector. The data is copied into the problem.
        @param lqrdata     A vector of LQR data. Each element is copied into the problem.
        @return 0 if successful
        """
        for k  in range(self.nhorizon):
            ndlqr_CopyLQRData(self.lqrdata[k], lqrdata[k])
        self.x0 = x0
        return 0
