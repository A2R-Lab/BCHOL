def PrintComp(base, new):
    print(f"{base} / {new} ({base / new} speedup)")

### Time only
class NdLqrProfile(object):
    """
    @brief A struct describing how long each part of the solve took, in milliseconds.

    ## Methods
    - ndlqr_NewNdLqrProfile() # Srishti - "unnecesaary -> this is init in Python"
    - ndlqr_ResetProfile()
    - ndlqr_CopyProfile()
    - ndlqr_PrintProfile()
    - ndlqr_CompareProfile()
    """
    def _init_(self):
        """
        @brief Create a profile initialized with zeros
        """
        self.t_total_ms = 0.0
        self.t_leaves_ms = 0.0
        self.t_products_ms = 0.0
        self.t_cholesky_ms = 0.0
        self.t_cholsolve_ms = 0.0
        self.t_shur_ms = 0.0
        self.num_threads = -1

    # Srishti - "Commenting this out because unnecesaary -> this is init in Python"
    '''
    def ndlqr_NewNdLqrProfile(self):    
        prof = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1]
        return prof
    '''

    def ndlqr_ResetProfile(self):
        """
        @brief Reset the profile to its initialized state

        @param prof A profile
        """
        self.t_total_ms = 0.0
        self.t_leaves_ms = 0.0
        self.t_products_ms = 0.0
        self.t_cholesky_ms = 0.0
        self.t_cholsolve_ms = 0.0
        self.t_shur_ms = 0.0
        return

    def ndlqr_CopyProfile(self, dest, src): # Srishti - "needs to be changed -> self is dest or src, depending on usage, currently no usage seen in Python code"
        """
        @brief Copy the profile information to a new profile

        @param dest New location for data. Existing data will be overwritten.
        @param src Data to be copied.
        """
        dest.num_threads = src.num_threads
        dest.t_total_ms = src.t_total_ms
        dest.t_leaves_ms = src.t_leaves_ms
        dest.t_products_ms = src.t_products_ms
        dest.t_cholesky_ms = src.t_cholesky_ms
        dest.t_cholsolve_ms = src.t_cholsolve_ms
        dest.t_shur_ms = src.t_shur_ms
        return

    def ndlqr_PrintProfile(self, profile):
        """
        @brief Print a summary fo the profile

        @param profile
        """
        print(f"Solved with {profile.num_threads} threads")
        print(f"Solve Total:    {profile.t_total_ms} ms")
        print(f"Solve Leaves:   {profile.t_leaves_ms} ms")
        print(f"Solve Products: {profile.t_products_ms} ms")
        print(f"Solve Cholesky: {profile.t_cholesky_ms} ms")
        print(f"Solve Solve:    {profile.t_cholsolve_ms} ms")
        print(f"Solve Shur:     {profile.t_shur_ms} ms")
        return

    def ndlqr_CompareProfile(self, base, prof): # Srishti - "needs to be changed -> self is base or prof, depending on usage, currently no usage seen in Python code"
        """
        @brief Compare two profiles, printing the comparison to stdout

        @param base The baseline profile
        @param prof The "new" profile
        """
        print(f"Num Threads:     {base.num_threads} / {prof.num_threads}")
        print(f"Solve Total:     ")
        PrintComp(base.t_total_ms, prof.t_total_ms)
        print(f"Solve Leaves:    ")
        PrintComp(base.t_leaves_ms, prof.t_leaves_ms)
        print(f"Solve Products:  ")
        PrintComp(base.t_products_ms, prof.t_products_ms)
        print(f"Solve Cholesky:  ")
        PrintComp(base.t_cholesky_ms, prof.t_cholesky_ms)
        print(f"Solve CholSolve: ")
        PrintComp(base.t_cholsolve_ms, prof.t_cholsolve_ms)
        print(f"Solve Shur Comp: ")
        PrintComp(base.t_shur_ms, prof.t_shur_ms)