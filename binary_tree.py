class BinaryNode (object):
    """
    @brief One of the nodes in the binary tree for the rsLQR solver
    """
    """
    #How do we initialize binaryNode? - Yana
    def _init_(self):
        self.idx # < knot point index
        self.level # < level in the tree
        self.levelidx # < leaf index at the current level
        self.left_inds # < range of knot point indices of all left children #do we need them? (Srishti - "idk")
        self.right_inds # < range of knot point indices of all right children
        self.parent # < parent node
        self.left_child # < left child
        self.right_child # < right child
    """
    #Not sure how do we initialize self.idx and self.parent?
    def _init_(self,start,len): 
        """
        actualy builds the tree,need to implement the function (Srishti - "implemented this now, but not sure if it's the correct translation from C")
        """
        mid = (len + 1) // 2
        new_len = mid - 1
        self.parent = None 
        self.idx = 0 #DOUBLE CHECK!
        if len > 1:
            self = start + new_len #binaryNode, how can we add BinaryNode* and int??
            left_root = BinaryNode(start, new_len)
            right_root = BinaryNode(start + mid, new_len)
            self.left_child = left_root
            self.right_child = right_root
            left_root.parent = self
            right_root.parent = self
            self.left_inds.start = left_root.left_inds.start
            self.left_inds.stop = left_root.right_inds.stop
            self.right_inds.start = right_root.left_inds.start
            self.right_inds.stop = right_root.right_inds.stop
            self.level = left_root.level + 1
            
        else:
            k = start.idx
            self.left_inds.start = k
            self.left_inds.stop = k
            self.right_inds.start = k + 1
            self.right_inds.stop = k + 1
            self.level = 0
            self.left_child = None
            self.right_child = None
            
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
    """
    def _init_(self,nhorizon): # (Srishti - "why does it have nhorizon as input"; 
    #Yana - "Because we need the length of the length of the time horizon to buils the tree")
        self.root # (C type: BinaryNode*) < root of the tree. Corresponds to the "middle" knot point.
        self.node_list = [] # (C type: BinaryNode*) < a list of all the nodes, ordered by their knot point index
        self.num_elements # (C type: int) < length of the OrderedBinaryTree::node_list
        self.depth # (C type: int) < total depth of the tree
    """
    #originally ndlqr_BuildTree

    def _init_(self, nhorizon): # (Srishti - "Is N (of C code) = nhorizon (of Python code)? - I think yes, look at .c file")
        """
        @brief Construct a new binary tree for a horizon of length @p N
        
        
        @param  N horizon length. Must be a power of 2.
        @return A new binary tree
        """
        assert(isPowerOfTwo(nhorizon)) #calling outside function here
        self.node_list = []
        for i in range (nhorizon):
            self.node_list[i].idx = i # (Srishti - "don't we have to make node_list of length `nhorizon` first?")
                                    # (Yana - "In python we don't have to specify the length of the list")
        self.num_elements = nhorizon
        self.depth = math.log2(nhorizon)
        
        # Build the tree
        self.root = BinaryNode(node_list, nhorizon - 1)

      
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

    #"Is there const functions in python?"
    def ndlqr_GetNodeAtLevel (node,index,level):
        if(node.level == level):
            return node
        elif(node.level>level):
            if(index <= node.idx):
                return GetNodeAtLevel (node.left_child, index, level)
            else:
                return GetNodeAtLevel(node.right_child, index, level)
        else:
            return GetNodeAtLevel(node.parent, index, level)

    def ndlqr_GetIndexAtLevel(self, leaf, level):
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
        if tree == None: # (Srishti - "is this syntax correct for Python" Yana -"Yep!")
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
