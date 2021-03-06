import numpy as np
import random
from anytree import NodeMixin, Node, RenderTree, LevelOrderIter
from anytree.exporter import DotExporter

def tree_grow(x, y, nfeat, nmin = 2, minleaf = 1):
    """
    Parameters:
        x (2D array): Data matrix
        y (1D array): Binary class labels
        nmin (int): Min # observations a node must contain for it to be allowed to split
        minleaf (int): Min # observations required for a leaf node
        nfeat (int): # features to be considered for each split

    Returns:
        Tree object based on best splits of gini index impurity reduction function
    """
    # each node has a name, list of indices (records), and "leaf" boolean attribute
    root = Node('root', 
                indices = np.arange(0, x.shape[0]), 
                leaf = False,
                prediction = None,
                split_feat = None,
                split_val = None)
    nodelist = [root]
    split_nr = 0 # will be used for node names
    while nodelist: # while nodelist not empty
        split_nr += 1
        current_node = nodelist.pop(0)  # get node from nodelist TODO: choose random node or first on list?
        print(f"\n PROCESSING NEW NODE {current_node} on subtree: x = \n {x[current_node.indices, :]} \n y = {y[current_node.indices]}")
        # TODO: skip this if nfeat not specified? Adjust optional nfeat in tree_grow def
        if nfeat:
            feat_list = random.sample(list(np.arange(0, x.shape[1])), k=nfeat)  # randomly draw nfeat col indices from # cols of x
        else:
            feat_list = list(np.arange(0, x.shape[1]))  # feat_list is simply indices of all columns of x (except first = indices)
        poss_splits = []  # will contain for each feature index (in col1), the impurity reduction (col2) & split value (col3) of the best split
        for f in feat_list:
            # find the best split (based on gini index) for rows of x specific by current_node.indices list, based on col f
            [reduction_val, split_val] = bestsplit_of_col(x[current_node.indices, f], y[current_node.indices], minleaf)
            if reduction_val != 0:  # if found a split which is allowed, then this is the best split for feature f
                poss_splits.append([f, reduction_val, split_val])
        if not poss_splits: # no possible split found
            current_node.leaf = True
            # add class prediction label to leaf node:
            if sum(current_node.indices)/len(current_node.indices) > 0.5:
                current_node.prediction = 1
            else:
                current_node.prediction = 0
        else: # choose split with highest impurity reduction:
            poss_splits.sort(key = lambda x: x[1], reverse = True) # sort poss_splits list by the 2nd column (reduction values) in descending order
            # TODO: add tiebraker in case of 2 features with identical impurity reduction -> keep list of features previsouly used for splitting?
            feat = poss_splits[0][0]
            split_val = poss_splits[0][2]
            current_node.split_feat = feat # add feature (col nr of x) and split value by which node will be split
            current_node.split_val = split_val
            # from indices in current nodes (current_node.indices), select those where value in column f > split_val
            indices_left = current_node.indices[x[current_node.indices,feat] > split_val]
            left = Node(f"L{split_nr}", parent=current_node, indices=indices_left)
            # if child node too small for splitting or we have a pure node (impurity=0), make it a leaf node:
            if ( len(indices_left) < nmin) or ( impurity(y[indices_left]) == 0):
                left.leaf = True
                left.y = y[indices_left]
                if sum( (left.y) / len(left.y) ) > 0.5:
                    left.prediction = 1
                else:
                    left.prediction = 0
            else: # make child node and add to nodelist
                left.leaf = False
                nodelist.append(left)
            indices_right = np.setdiff1d(current_node.indices, indices_left)  # indices_right = indices in current node not in indices_left
            right = Node(f"R{split_nr}", parent=current_node, indices=indices_right)
            if ( len(indices_right) < nmin) or ( impurity(y[indices_right]) == 0 ): # make child leaf node
                right.leaf = True
                right.y = y[indices_right]
                if ( sum(right.y) / len(right.y)) > 0.5:
                    right.prediction = 1
                else:
                    right.prediction = 0
            else: # make child node and add to nodelist
                right.leaf = False
                nodelist.append(right)
            print(f"Finished processing node, tree now looks as follows: \n {RenderTree(root)}")
    print(f"\n TREE DONE: \n {RenderTree(root)}")
    return root

def tree_pred(x, tr):
    """
    Parameters:
        x (2D array): Attribute data matrix 
        tr (AnyTree): Clasification tree object

    Returns:
        List of predicted labels.
    """
    y = list()
    for i in range(len(x)):
        node_list = [node for node in LevelOrderIter(tr)]
        for node in node_list:
            if node.leaf and i in node.indices:
                y.append(node.prediction)
                break
            elif not node.leaf and node.split_val > x[i][node.split_feat]:
                node_list = [node for node in LevelOrderIter(node)]
    return y

def impurity(x):
    """
    Parameters:
        x: Input binary vector of class labels

    Returns:
        Impurity of that node according to Gini index function
    """
    n = len(x) # records in node
    impurity = sum(x)*(n-sum(x))/(n**2)
    return impurity

def impurity_reduction(parent, left_child, right_child):
    """
    Parameters:
        binary class label vectors of parent node
        2 child nodes of possible split

    Returns:
        Impurity reduction value of that split
    """
    impurity_parent = impurity(parent)
    impurity_l = impurity(left_child)
    impurity_r = impurity(right_child)
    imp_red = impurity_parent - ((len(left_child)/len(parent))*impurity_l + (len(right_child)/len(parent))*impurity_r)
    return imp_red

def bestsplit_of_col(x, y, minleaf):
    """
    Parameters:
        x: numeric attribute vector
        y: class label vector
        minleaf: minimum size allowed for leaf node

    Returns:
        Best split (highest impurity reduction) & split value
    """
    x_sorted = np.sort(np.unique(x))    #sort x in increasing order, with only unique values
    if len(x_sorted) == 1: # All values of vector x are identical
        return [0,0]
    splitpoints = (x_sorted[0:len(x_sorted)-1]+x_sorted[1:len(x_sorted)])/2
    # try all these possible splits:
    max_imp_red = 0
    best_split_val = splitpoints[0]
    for s in list(splitpoints):
        indices_left = np.arange(0, len(x)) [x > s] # take row index of elements in x with value > s
        indices_right = np.delete(np.arange(0, len(x)), indices_left)  # make list of indices & remove those in indices_left
        if (len(indices_left) < minleaf) or (len(indices_right) < minleaf):
            print(f"Child node {indices_left} or {indices_right} would be too small; no split allowed")
            continue
        left_child = y[indices_left]
        right_child = y[indices_right]
        imp_red = impurity_reduction(y, left_child, right_child)
        if imp_red > max_imp_red:
            max_imp_red = imp_red
            best_split_val = s
    return [max_imp_red, best_split_val]

# TODO: maybe split up tree_grow function, adding function below which finds the best split,
#  and adding another function which performs the best split
'''
def multivariate_best_splits(x, y, current_node, minleaf, feat_list):
    # input: data matrix x, class label vector y, minleaf, and featlist = list of columns of x to consider for split
    # output: array with each row = [f, imp_red, split_val]
    # where f = feature index, imp_red = impurity reduction, split_val = value to split for best impurity reduction of that feature
    poss_splits = []  # will contain for each feature index (in col1), the impurity reduction (col2) & split value (col3) of the best split
    for f in feat_list:
        # print(f"Finding best split for column feature {f}")
        # find the best split (based on gini index) for rows of x specific by current_node.indices list, based on col f
        [reduction_val, split_val] = bestsplit_of_col(x[current_node.indices, f], y[current_node.indices], minleaf)
        if reduction_val != 0:  # if found a split which is allowed, then this is the best split for feature f
            poss_splits.append([f, reduction_val, split_val])
'''

def nodeattrfunc(node):
    if node.leaf:
        return f'label = "LEAF {node.name}:\n ind = {node.indices} \n labels = {node.y} \n prediction = {node.prediction}", shape="diamond"'
    else:
        return f'label = "Node {node.name}:\n ind = {node.indices}"'    # works!

def edgeattrfunc(parent, child):
    if 'L' in child.name: # we have a left child
        return f'label= "x[:,{parent.split_feat}] > {parent.split_val}"'
    else: # we have a right child
        return f'label= "x[:,{parent.split_feat}] \u2264 {parent.split_val}"' # \u2264 is python source code for <=


credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
x = credit_data[:,0:5]
y = credit_data[:,5]
#pima = np.genfromtxt('pima.txt', delimiter=',', skip_header=False)
#x = pima[0:100,0:pima.shape[1]-1]
#y = pima[0:100,pima.shape[1]-1]
tree = tree_grow(x, y, nfeat=x.shape[1], nmin = 2, minleaf=1)
print(y)
print(tree_pred(x, tree))
DotExporter(tree).to_picture("tree_only.png") # works!
DotExporter(tree, nodeattrfunc=nodeattrfunc, edgeattrfunc=edgeattrfunc).to_picture("credit_tree.png")



## Testing functions

'''
y= np.array([1,0,1,1,1,0,0,1,1,0,1])
print(impurity(y))

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
[v,s] = bestsplit(credit_data[:,3], credit_data[:,5], 2)
print(s)
print(v)

y= np.array([1,0,1,1,1,0,0,1,1,0,1])
root = Node('root', indices = np.arange(0,len(y)), y = y)
root.impurity = impurity(y)
ind_L = [0,1,2,3]
L1 = Node('L1', indices = ind_L, y = y[ind_L], parent=root)
ind_R = np.delete(np.arange(0,len(y)), ind_L)
R1 = Node('R1', indices = ind_R, y = y[ind_R], parent = root)
'''