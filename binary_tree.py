class BinaryNode (object):
    """
    @brief One of the nodes in the binary tree for the rsLQR solver
    """
    def _init_(self):
        self.idx # < knot point index
        self.level # < level in the tree
        self.levelidx # < leaf index at the current level
        self.left_inds # < range of knot point indices of all left children #do we need them? (Srishti - "idk")
        self.right_inds # < range of knot point indices of all right children
        self.parent # < parent node
        self.left_child # < left child
        self.right_child # < right child

def BuildSubTree(start,len):
        """
        actualy builds the tree,need to implement the function (Srishti - "implemented this now, but not sure if it's the correct translation from C")
        """
        mid = (len + 1) // 2
        new_len = mid - 1
        if len > 1:
            root = start + new_len #binaryNode
            left_root = BuildSubTree(start, new_len)
            right_root = BuildSubTree(start + mid, new_len)
            root.left_child = left_root
            root.right_child = right_root
            left_root.parent = root
            right_root.parent = root
            root.left_inds.start = left_root.left_inds.start
            root.left_inds.stop = left_root.right_inds.stop
            root.right_inds.start = right_root.left_inds.start
            root.right_inds.stop = right_root.right_inds.stop
            root.level = left_root.level + 1
            return root
        else:
            k = start.idx
            start.left_inds.start = k
            start.left_inds.stop = k
            start.right_inds.start = k + 1
            start.right_inds.stop = k + 1
            start.level = 0
            start.left_child = None
            start.right_child = None
            return start

class OrderedBinarytree(object):
    """
    @brief The binary tree for the rsLQR solver
    
    Caches useful information to speed up some of the computations during the
    rsLQR solver, mostly around converting form linear knot point indices to
    the hierarchical indexing of the algorithm.
    
    ## Construction and destruction
    A new tree can be constructed solely given the length of the time horizon, which must
    be a power of two. Use ndlqr_BuildTree(N) to build a new tree, 
    """
    def _init_(self,nhorizon): # (Srishti - "why does it have nhorizon as input"; 
    #Yana - "Because we need the length of the length of the time horizon to buils the tree")
        self.root # (C type: BinaryNode*) < root of the tree. Corresponds to the "middle" knot point.
        self.node_list # (C type: BinaryNode*) < a list of all the nodes, ordered by their knot point index
        self.num_elements # (C type: int) < length of the OrderedBinaryTree::node_list
        self.depth # (C type: int) < total depth of the tree

    
    def ndlqr_BuildTree(self, nhorizon): # (Srishti - "Is N (of C code) = nhorizon (of Python code)? - I think yes, look at .c file")
        """
        @brief Construct a new binary tree for a horizon of length @p N
        
        
        @param  N horizon length. Must be a power of 2.
        @return A new binary tree
        """
        assert(isPowerOfTwo(nhorizon))

        for i in range (nhorizon):
            self.node_list[i].idx = i # (Srishti - "don't we have to make node_list of length `nhorizon` first?")
        self.num_elements = nhorizon
        self.depth = math.log2(nhorizon)
        
        # Build the tree
        self.root = BuildSubTree(node_list, nhorizon - 1)

    #don't need ndlqr_FreeeTree (Srishti - "why not?")
      
    def ndlqr_GetIndexFromLeaf(self, leaf, level):
        """
        @brief Get the knot point index given the leaf index at a given level

        @param tree  An initialized binary tree for the problem horizon
        @param leaf  Leaf index
        @param level Level of tree from which to get the index
        @return
        """
        linear_index = math.pow(2, level) * (2 * leaf + 1) - 1
        return linear_index
    
    def ndlqr_GetIndexLevel(self, index):
        """
        @brief Get the level for a given knot point index

        @param tree  An initialized binary tree for the problem horizon
        @param index Knot point index
        @return      The level for the given index
        """
        return self.node_list[index].level   

    def ndlqr_GetIndexAtLevel(tree, leaf, level):
        """
        @brief Get the index in 'level' that corresponds to `index`.

        If the level is higher than the level of the given index, it's simply the parent
        at that level. If it's lower, then it's the index that's closest to the given one, with
        ties broken by choosing the left (or smaller) of the two.

        @param tree  Precomputed binary tree
        @param index Start index of the search. The result will be the index closest to this
        index.
        @param level The level in which the returned index should belong to.
        @return int  The index closest to the provided one, in the given level. -1 if
        unsucessful.
        """
        if tree = None: # (Srishti - "is this syntax correct for Python")
            return -1
        if index < 0 or index >= tree.num_elements:
            print(f"ERROR: Invalid index ({index}). Should be between 0 and {tree.num_elements - 1}.")
        
        if level < 0 or level >= tree.depth:
            print(f"ERROR: Invalid level ({level}). Should be between 0 and {tree.depth - 1}.")

        node = tree.node_list + index
        if index == tree.num_elements - 1:
            node = tree.node_list + index - 1

        base_node = GetNodeAtLevel(node, index, level) # (Srishti - "check how to define const variable here (syntax)")
        return base_node.idx

def PrintComp(base, new):
    print(f"{base} / {new} ({base / new} speedup)")

class NdLqrProfile(object):
    """
    @brief A struct describing how long each part of the solve took, in milliseconds.

    ## Methods
    - ndlqr_NewNdLqrProfile()
    - ndlqr_ResetProfile()
    - ndlqr_CopyProfile()
    - ndlqr_PrintProfile()
    - ndlqr_CompareProfile()
    """
    def _init_(self):
        self.t_total_ms
        self.t_leaves_ms
        self.t_products_ms
        self.t_cholesky_ms
        self.t_cholsolve_ms
        self.t_shur_ms
        self.num_threads

    def ndlqr_NewNdLqrProfile(self):
        """
        @brief Create a profile initialized with zeros
        """
        prof = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1]
        return prof

    def ndlqr_ResetProfile(self, prof):
        """
        @brief Reset the profile to its initialized state

        @param prof A profile
        """
        prof.t_total_ms = 0.0
        prof.t_leaves_ms = 0.0
        prof.t_products_ms = 0.0
        prof.t_cholesky_ms = 0.0
        prof.t_cholsolve_ms = 0.0
        prof.t_shur_ms = 0.0
        return

    def ndlqr_CopyProfile(self, dest, src):
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

    def ndlqr_CompareProfile(self, base, prof):
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