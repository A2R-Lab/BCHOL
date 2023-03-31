class NdFactor(object):
    """
    @brief A chunk of memory for a single time step

    Stores a matrix of size (2n+m,n), divided into blocks:

    \f[
    \begin{bmatrix} \Lambda \\ X \\ U \end{bmatrix}
    \f]

    which correspond to the NdFactor.lambda, NdFactor.state, and NdFactor.input
    fields, which can also be extracted using the following methods:
    - ndlqr_GetLambdaFactor()
    - ndlqr_GetStateFactor()
    - ndlqr_GetInputFactor()

    Each of which return a Matrix of the corresponding size. Each of these blocks can have
    an arbitrary width, since these can represent either data from the KKT matrix or the
    right-hand-side vector(s).

    Internally, the solver stores arrays of these objects, which allow the solver
    to extract chunks out of the original matrix data, by time step.
    """
    def _init_(self): # Srishti - how is this done??
        self.lambda # Srishti - check if lambda is a keyword < (n,w) block for the dual variables
        self.state # < (n,w) block for the state variables
        self.input # < (m,w) block for the control input variables

    def ndlqr_GetLambdaFactor(self):
        return self.lambda

    def ndlqr_GetStateFactor(self):
        return self.state

    def ndlqr_GetInputFactor(self):
        return self.input

class NdData(object):
    """
    @brief Core storage container for the rsLQR solver

    Represents an array of memory blocks, arranged as follows:

    \f[
    \begin{bmatrix}
    F_1^{(1)} & X_1^{(2)} & \dots & X_1^{(K)} \\
    F_2^{(1)} & X_2^{(2)} & \dots & X_2^{(K)} \\
    \vdots    & \vdots    & \ddots & \vdots \\
    F_{N-1}^{(1)} & X_{N-1}^{(2)} & \dots & X_{N-1}^{(K)} \\
    \end{bmatrix}
    \f]

    Each \f$ F \f$ is a NdFactor further dividing this memory into chunks of size `(n,w)` or
    `(m,w)`, where the width `w` is equal to NdData.width. Each block is stored as an
    individual Matrix, which stores the data column-wise. This keeps the data for a single
    block together in one contiguous block of memory. The entire block of memory for all of
    the factors is allocated as one large block (with pointer NdData.data).

    In the solver, this is used to represent both the KKT matrix data and the right-hand-side
    vector. When storing the matrix data, each column represents a level of the binary tree.
    The current implementation only allows for a single right-hand-side vector, so that
    when `width` is passed to the initializer, it only creates a single column of factors.
    Future modifications could alternatively make the right-hand-side vector the last column
    in the matrix data, as suggested in the original paper.

    ## Methods
    - ndlqr_NewNdData()
    - ndlqr_FreeNdData()
    - ndlqr_GetNdFactor()
    - ndlqr_ResetNdFactor()
    """
    def _init_(self):
        self.nstates    #< size of state vector
        self.ninputs    #< number of control inputs
        self.nsegments  #< number of segments, or one less than the length of the horizon
        self.depth      #< number of columns of factors to store
        self.width      #< width of each factor. Will be `n` for matrix data and typically 1 for the right-hand-side vector.
        self.data       #< pointer to entire chunk of allocated memory
        self.factors  #< (nsegments, depth) array of factors. Stored in column-order.

    def ndlqr_NewNdData(self, nstates, ninputs, nhorizon, width):
        """
        @brief Initialize the NdData structure

        Note this allocates a large block of memory. This should be followed by a single
        call to ndlqr_FreeNdData().

        @param nstates Number of variables in the state vector
        @param ninputs Number of control inputs
        @param nhorizon Length of the time horizon
        @param width With of each factor. Should be `nstates` for KKT matrix data,
                    or 1 for the right-hand side vector.
        @return The initialized NdData structure
        """
        nsegments = nhorizon - 1
        if nstates <= 0 or ninputs <= 0 or nsegments <= 0:
            return None
        if not IsPowerOfTwo(nhorizon):
            print("ERROR: Number of segments must be one less than a power of 2.")
            return None

        # A little hacky, but set depth to 1 for the rhs vector
        depth = 0
        if width == 1:
            depth = 1
        else:
            depth = LogOfTwo(nhorizon)

        # Allocate one large block of memory for the data
        numfactors = nhorizon * depth
        factorsize = (2 * nstates + ninputs) * width
        data = # Srishti - "how to do this?"

        # Create the factors using the allocated memory
        factors = # Srishti - "how to do this?"
        for i in range(numfactors):
            factordata = data + i * factorsize
            factors[i].lambda.rows = nstates
            factors[i].lambda.cols = width
            factors[i].lambda.data = factordata
            factors[i].state.rows = nstates
            factors[i].state.cols = width
            factors[i].state.data = factordata + nstates * width
            factors[i].input.rows = ninputs
            factors[i].input.cols = width
            factors[i].input.data = factordata + 2 * (nstates * width)

        # Create the NdData struct
        nddata = # Srishti - "how to do this?"
        nddata.nstates = nstates
        nddata.ninputs = ninputs
        nddata.nsegments = nsegments
        nddata.depth = depth
        nddata.width = width
        nddata.data = data
        nddata.factors = factors
        return nddata

    def ndlqr_ResetNdData(self, nddata):
        """
        @brief Resets all of the memory for an NdData to zero.

        @param nddata Initialized NdData structure
        """
        nhorizon = nddata.nsegments + 1
        numfactors = nhorizon * nddata.depth
        factorsize = (2 * nddata.nstates + nddata.ninputs) * nddata.width
        for i in range(numfactors * factorsize): # Srishti - "verify whether this is correct"
            nddata.data[i] = 0

    def ndlqr_GetNdFactor(self, nddata, index, level, factor):
        """
        @brief Retrieve an individual NdFactor out of the NdData

        Retrieves a block of memory out of NdData, stored as an NdFactor. Typical usage
        will look like:

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        NdData* ndata = ndlqr_NewNdData(nstates, ninputs, nhorizon, nstates);
        NdFactor* factor;
        int index = 0;  // set to desired index
        int level = 1;  // set to desired level
        ndlqr_GetNdFactor(nddata, index, level, &factor);
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        @param nddata Storage location for the desired block of memory
        @param index Time step of the factor to extract
        @param level Level (or column in the NdData) of the desired factor
        @param factor Storage location for the factor.
        @return 0 if successful
        """
        if index < 0 or index > nddata.nsegments:
            print(f"Invalid index. Must be between {0} and {nddata.nsegments}, got {index}.")
            return -1

        if level < 0 or level >= nddata.depth:
            print(f"Invalid level. Must be between {0} and {nddata.depth - 1}, got {level}.")
            return -1

        linear_index = index + (nddata.nsegments + 1) * level
        factor = nddata.factors + linear_index
        return 0
