class BinaryNode (object):
    """
    @brief One of the nodes in the binary tree for the rsLQR solver
    """
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
    """
    def __init__(self):
        self.parent = None
        self.idx = 0
        self.left_child = None
        self.right_child = None
        self.left_inds_start = -1
        self.left_inds_stop = -1
        self.right_inds_start = -1
        self.right_inds_stop = -1
        self.level = 0
        

def buildSubTree(start, len):
    """
    actualy builds the tree
    """
    mid = (len + 1) // 2
    new_len = mid - 1
    current = BinaryNode()
    if len > 1:
        current = start[new_len]
        left_root = buildSubTree(start[0:], new_len)
        right_root = buildSubTree(start[mid:], new_len)
        current.left_child = left_root
        current.right_child = right_root
        left_root.parent = current
        right_root.parent = current
        current.left_inds_start = left_root.left_inds_start
        current.left_inds_stop = left_root.right_inds_stop
        current.right_inds_start = right_root.left_inds_start
        current.right_inds_stop = right_root.right_inds_stop
        current.level = left_root.level + 1
    else:
        k = start[0].idx
        current.left_inds_start = k
        current.left_inds_stop = k
        current.right_inds_start = k + 1
        current.right_inds_stop = k + 1
        current.level = 0
        current.left_child = None
        current.right_child = None
    return current

class OrderedBinaryTree(object):
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

    def __init__(self, nhorizon):
        """
        @brief Construct a new binary tree for a horizon of length @p N
        
        
        @param  N horizon length. Must be a power of 2.
        @return A new binary tree
        """
        assert(isPowerOfTwo(nhorizon)) #calling outside function here
        self.node_list = [] #BinaryNode list
        for i in range (nhorizon):
            bn = BinaryNode()
            bn.idx = i
            self.node_list.append(bn)
        self.num_elements = nhorizon
        self.depth = math.log2(nhorizon)
        
        # Build the tree
        self.root = buildSubTree(self.node_list, nhorizon - 1)

    # QUESTION : WHERE DO WE ACTUALLY ACCESSING THE TREE IN THIS FUNCTION?
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

def GetNodeAtLevel (node,index,level):
  if(node.level == level):
    return node
  elif(node.level>level):
    if(index <= node.idx):
      return GetNodeAtLevel (node.left_child, index, level)
    else:
      return GetNodeAtLevel(node.right_child, index, level)
  else:
      return GetNodeAtLevel(node.parent, index, level)


#I think it should also be outside of class
def ndlqr_GetIndexAtLevel(tree, index, level):
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
  if (tree == None):
    return -1
  if index < 0 or index >= tree.num_elements:
    print(f"ERROR: Invalid index ({index}). Should be between 0 and {tree.num_elements - 1}.")
        
  if level < 0 or level >= tree.depth:
    print(f"ERROR: Invalid level ({level}). Should be between 0 and {tree.depth - 1}.")

  node = tree.node_list + index
  if index == tree.num_elements - 1:
    node = tree.node_list + index - 1

  base_node = GetNodeAtLevel(node, index, level)
  return base_node.idx
