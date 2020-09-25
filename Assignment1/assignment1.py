import numpy as np
import random
from anytree import NodeMixin, Node, RenderTree
from anytree.exporter import DotExporter

def tree_grow(x, y, nfeat, nmin = 2, minleaf = 1):
    """
    Input parameters:
        x (2D array): Data matrix
        y (1D array): Binary class labels
        nmin (int): Min # observations a node must contain for it to be allowed to split
        minleaf (int): Min # observations required for a leaf node
        nfeat (int): # features to be considered for each split
    Outputs:
        Tree object based on best splits of gini index impurity reduction function
    """
    print("GROWING CLASSIFICATION TREE")
    # each node has a name, list of indices (records), and "leaf" boolean attribute
    root = Node('root', indices = np.arange(0, x.shape[0]), leaf = False)
    nodelist = [root]
    split_nr = 0 # will be used for node names
    while nodelist: # while nodelist not empty
        split_nr += 1
        current_node = nodelist.pop(0)  # get node from nodelist TODO: choose random node or first on list?
        #print(f"\n PROCESSING NEW NODE {current_node}")# on subtree: x = \n {x[current_node.indices, :]} \n y = {y[current_node.indices]}")
        # TODO: skip this if nfeat not specified? Adjust optional nfeat in tree_grow def
        if nfeat:
            feat_list = random.sample(list(np.arange(0, x.shape[1])), k=nfeat)  # randomly draw nfeat col indices from # cols of x
        else:
            feat_list = list(np.arange(0, x.shape[1]))  # feat_list is simply indices of all columns of x (except first = indices)
        poss_splits = []  # will contain for each feature index (in col1), the impurity reduction (col2) & split value (col3) of the best split
        for f in feat_list:
            # print(f"Finding best split for column feature {f}")
            # find the best split (based on gini index) for rows of x specific by current_node.indices list, based on col f
            [reduction_val, split_val] = bestsplit_of_col(x[current_node.indices, f], y[current_node.indices], minleaf)
            if reduction_val != 0:  # if found a split which is allowed, then this is the best split for feature f
                poss_splits.append([f, reduction_val, split_val])
        #print(f"Finished finding the best splits for each feature, poss_splits = \n {poss_splits}")
        if not poss_splits: # no possible split found
            current_node.leaf = True
            # add class prediction label to leaf node:
            current_node.y = y[current_node.indices]
            if sum((current_node.y) / len(current_node.y)) > 0.5:
                current_node.prediction = 1
            else:
                current_node.prediction = 0
        else: # choose split with highest impurity reduction:
            poss_splits.sort(key = lambda x: x[1], reverse = True) # sort poss_splits list by the 2nd column (reduction values) in descending order
            # TODO: add tiebraker in case of 2 features with identical impurity reduction -> look up previous features via parent?
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
            else: # add to nodelist
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
            else: # add to nodelist
                right.leaf = False
                nodelist.append(right)
            #print(f"Finished processing node, tree now looks as follows: \n {RenderTree(root)}")
    #print(f"TREE DONE")#\n {RenderTree(root)}")
    return root

def tree_pred(x, tr):
    """
   Input parameters:
       x (2D array): Attribute data matrix
       tr (AnyTree): Classification tree object
   Outputs:
       List of predicted labels.
   """

    print(f"TREE_PRED started")#for x = \n {x}")
    n_rows = x.shape[0] # number of rows
    #print(f"matrix x {x.shape} has n_rows = {n_rows}")
    y = np.zeros(n_rows)
    for i in np.arange(0, n_rows): # for each row = record in x, go down tree
        #print(f"\n PROCESSING new row {x[i,:]} of x")
        node = tr # start at root node
        while not node.leaf: # repeat until we have reached a leaf node
            #print(f"Not a leaf node, so splitting from \n {node}")
            if x[i, node.split_feat] > node.split_val: # go to left child node
                node = node.children[0] #go to left child
                #print(f"Going in LEFT child node")
            else:
                node = node.children[1] #go to right child
                #print(f"Going into RIGHT child node")
        y[i] = node.prediction
        #print(f"Found leaf node {node}, predicting {y[i]}")
    #print("FINISHED with tree_pred")
    return y

def tree_grow_b(x, y, m, nfeat, nmin = 2, minleaf = 1):
    """
    Input parameters:
        x (2D array): Data matrix
        y (1D array): Binary class labels
        m (int): number of bootstrapped trees to be made
        nmin (int): Min # observations a node must contain for it to be allowed to split
        minleaf (int): Min # observations required for a leaf node
        nfeat (int): # features to be considered for each split
    Outputs:
        List of tree objects made from bootstrap samples, each based on best splits of gini index impurity reduction function
    """
    print(f"STARTING TREE_GROW_B, making {m} bootstrap samples and growing a new tree for each sample")
    trees = [] # list will contain m trees grown from bootstrap samples
    i=0
    while (i != m):
        i+=1
        print(f"Bootstrap {i}")
        n_samples = x.shape[0]
        ind = np.random.randint(n_samples, size=n_samples)  # list of indices for bootstrap sample
        sample_x = x[ind, :]
        sample_y = y[ind]
        tree = tree_grow(sample_x, sample_y, nfeat, nmin, minleaf)
        trees.append(tree)
    return trees


def tree_pred_b(x, trees):
    """
    Input parameters:
        x (2D array): Attribute data matrix
        trees: list of tree objects
    Outputs:
        list of predicted labels obtained by majority vote of predictions from trees
    """
    n_trees = len(trees) # number of bootstrapped trees
    y_predictions = []
    for tree in trees:
        y_pred = tree_pred(x,tree)
        #print(f"New prediction = {y_pred}")
        y_predictions.append(y_pred)
    y_predictions = np.array(y_predictions) # convert list to array
    #print(f"Prediction matrix = \n {y_predictions}")
    # take mean for each column of y_predictions, and see whether mean < 0.5 -> assign prediction = 0
    final_pred = list(map(lambda v: 0 if v < 0.5 else 1, np.sum(y_predictions, axis=0)/n_trees))
    #print(f"final_pred = {final_pred}")
    return final_pred

def impurity(x):
    """
    Input parameter:
        x: binary vector of class labels
    Outputs:
        Impurity of that node according to Gini index function
    """
    n = len(x) # records in node
    impurity = sum(x)*(n-sum(x))/(n**2)
    return impurity

def impurity_reduction(parent, left_child, right_child):
    """
    Input parameters:
        parent: left_child, right_child: binary class label vectors of parent node and of 2 child nodes of possible split
    Outputs:
        Impurity reduction value of that split
    """
    impurity_parent = impurity(parent)
    #print(f"\nComputing impurity reduction of splitting parent {parent} = {impurity_parent}")
    impurity_l = impurity(left_child)
    impurity_r = impurity(right_child)
    #print(f"into left child {left_child} with impurity {impurity_l} and \n right child {right_child} with impurity {impurity_r}")
    imp_red = impurity_parent - ((len(left_child)/len(parent))*impurity_l + (len(right_child)/len(parent))*impurity_r)
    #print(f"reduction = {imp_red}")
    return imp_red

def bestsplit_of_col(x, y, minleaf):
    """
    Input parameters:
        x: numeric attribute vector
        y: class label vector
        minleaf: minimum size allowed for leaf node
    Outputs:
        Best split (highest impurity reduction) & split value
    """
    x_sorted = np.sort(np.unique(x))    #sort x in increasing order, with only unique values
    #print(f"Finding the best split of numeric attribute vector: {x} for the class label vector: {y}")
    #print(f"sorted vector: \n {x_sorted}")
    if len(x_sorted) == 1: # All values of vector x are identical
        #print(f"All values of vector x identical so no split possible on this attribute")
        return [0,0]
    splitpoints = (x_sorted[0:len(x_sorted)-1]+x_sorted[1:len(x_sorted)])/2
    #print(f"splitpoints vec: \n {splitpoints}")
    # try all these possible splits:
    max_imp_red = 0
    best_split_val = splitpoints[0]
    for s in list(splitpoints):
        #print(f"Testing split on s={s}")
        indices_left = np.arange(0, len(x)) [x > s] # take row index of elements in x with value > s
        indices_right = np.delete(np.arange(0, len(x)), indices_left)  # make list of indices & remove those in indices_left
        if (len(indices_left) < minleaf) or (len(indices_right) < minleaf):
            #print(f"Child node {indices_left} or {indices_right} would be too small; no split allowed")
            continue
        left_child = y[indices_left]
        #print(f"indices_left = {indices_left}")
        right_child = y[indices_right]
        #print(f"indices right = {indices_right}")
        imp_red = impurity_reduction(y, left_child, right_child)
        if imp_red > max_imp_red:
            #print(f"updated split from {best_split_val} to {s} increasing the reduction value from {max_imp_red} to {imp_red}")
            max_imp_red = imp_red
            best_split_val = s
    #print(f"Best split at value {best_split_val} with impurity reduction {max_imp_red}")
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
    """
    Input parameter: node
    Outputs: string for labeling that node in the tree picture
    """
    if node.leaf:
        return f'label = "LEAF {node.name}:\n ind = {node.indices} \n labels = {node.y} \n prediction = {node.prediction}", shape="diamond"'
    else:
        return f'label = "Node {node.name}:\n ind = {node.indices}"'    # works!

def edgeattrfunc(parent, child):
    """
    Input parameters: parent and child node
    Outputs: string to label edge between these nodes in the tree picture
    """
    if 'L' in child.name: # we have a left child
        return f'label= "x[:,{parent.split_feat}] > {parent.split_val}"'
    else: # we have a right child
        return f'label= "x[:,{parent.split_feat}] \u2264 {parent.split_val}"' # \u2264 is python source code for <=

def confusion_matrix(y_true,y_pred):
    """
    Input parameters:
        y_true: binary vector of true class labels
        y_pred: binary vector of predicted labels
    Outputs: (2x2 array) confusion matrix
    """
    T = np.array(y_true,dtype=int)
    P = np.array(y_pred,dtype=int)
    cM = np.zeros((2,2))
    print(f"MAKING CONFUSION MATRIX ")  # for \n {y_pred} = predicted, \n {y_true} = true labels")
    for i in range(len(y_true)):
        cM[T[i],P[i]] += 1
    return cM


credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
x_train = credit_data[0:8,0:5]
y_train = credit_data[0:8,5]
x_test = credit_data[8:10,0:5]
y_test = credit_data[8:10, 5]
#tree = tree_grow(x_train, y_train, nfeat=x_train.shape[1], nmin = 2, minleaf=1)

'''


pima = np.genfromtxt('pima.txt', delimiter=',', skip_header=False)
x_train = pima[0:400,0:8]
y_train = pima[0:400, 8]
x_test = pima[400:767, 0:8]
y_test = pima[400:767, 8]
# testing as in assignment to compare confusion matrices:
#x_train = pima[:,0:8]
#y_train = pima[:,8]
#x_test = x_train
#y_test = y_train
'''

trees = tree_grow_b(x_train, y_train, 5, nfeat=x_train.shape[1], nmin = 20, minleaf=5)
y_pred = tree_pred_b(x_test, trees)
C = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix for with bootstrapped samples = \n {C}")

'''
tree = tree_grow(x_train, y_train, nfeat=x_train.shape[1], nmin = 20, minleaf=5)
y_pred  = tree_pred(x_test, tree)
#print(f"For x = \n {x_test} \n {y_pred} = tree predicted labels  \n {y_test} = true labels")
C = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix with just 1 tree (without bootstrapping) = \n {C}")



tree = tree_grow(x_train, y_train, nfeat=x_train.shape[1], nmin = 20, minleaf=5)

y_pred  = tree_pred(x_test, tree)
#print(f"For x = \n {x_test} \n {y_pred} = tree predicted labels  \n {y_test} = true labels")
C = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix for prediction = \n {C}")
#DotExporter(tree).to_picture("tree_only2.png") # works!
#DotExporter(tree, nodeattrfunc=nodeattrfunc, edgeattrfunc=edgeattrfunc).to_picture("credit_tree2.png")
'''