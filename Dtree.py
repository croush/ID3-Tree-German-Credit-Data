import random
import math
import pandas
from sklearn import cross_validation

#this is more like expected entropy since it's passed a column
def expectedEntropy(tCol): #it's passed a column and entropy (E) of the whole set's target column
                                # so that it can return E-sum=expected entropy
    sum = 0.0

    priors = []     #to hold the priors (used in entropy comp later)
    temp = tCol.value_counts()
    total=0.0                      #get the counts of each unique
    for i in temp.index:        #for each unique value, calculate the prior
        priors.append(temp.ix[i]/len(tCol.index))
        total += temp.ix[i]
     #   sum = sum + (-priors[i]*math.log(priors[i],2))

    for i in range(len(priors)):
        sum = sum + (-priors[i] * math.log(priors[i], 2))*priors[i]/total
    print(total)
    return sum;

def entropy(dataSet):
    sum = 0.0

    priors = []  # to hold the priors (used in entropy comp later)
    temp = dataSet['Creditability'].value_counts()
    for i in temp.index:  # for each unique value, calculate the prior
        priors.append(temp.ix[i]/len(dataSet.index))

    for i in range(len(priors)):
        sum = sum + (-priors[i] * math.log(priors[i], 2)) #need to do weighted calculations here
    return sum


# the class with functions to be used as entry points when
# either training (fit) or predicting (predict) with the
# decision tree algorithm
class DTree:
    def fit(self, predictor_columns_data, target_column_data):
        self.__root_node = DNode(predictor_columns_data, target_column_data)
        self.__root_node.train()

    def predict(self, df_of_new_examples):
        # apply the predict function to the whole series, one at a time, this returns the series with the return vals
        predictions = df_of_new_examples.apply(self.__root_node.predict, axis=1)
        return predictions

    def print_tree(self):
        self.__root_node.print_node()


# A class for representing non-leaf nodes in the decision tree
class DNode:
    # when we create this node, we pass it training examples to be used at this point
    # the predictor columns of these training examples is in predictor_columns_data
    # the corresponding target values to those predictor columns are in target_column_data
    def __init__(self, predictor_columns_data, target_column_data):

        self.__attribute = ''  # the attribute used to sort examples at this node
        self.__predictor_columns = predictor_columns_data  # the training examples that have been sorted to this node
        self.__target_column = target_column_data  # the corresponding target values for the training examples
        self.__child_nodes = {}  # dictionary of the child nodes of this node, indexed by the value they have for self.__attribute
        self.__most_common_value_here = ''  # for keeping track of which target value is most common among the examples at this node. This is used to make a decision when there's no appropriate child node to follow

    # this should use the training data to determine the best attribute to use
    # as is, it just chooses one at random, but you will fix it to use information gain
    def choose_attribute(self):
        self.__attribute = random.choice(
            self.__predictor_columns.columns.values)  # what a terrible way to choose the attribute!

    # calling this will continue building the tree from this node given its training examples
    def train(self):
        self.choose_attribute()  # 'best' attribute at this node

        # in case we need to make a decision here because we don't have any children with a particular attribute value
        self.__most_common_value_here = self.__target_column.value_counts().idxmax()

        # gets all the values that these examples have in our chosen column
        attribute_values_here = self.__predictor_columns[self.__attribute].unique()

        # going through all possible values this attribute can have
        # and creating the appropriate child node
        for value in attribute_values_here:

            # the subset of examples with the given value
            examples_for_child_predictor_cols = self.__predictor_columns[
                self.__predictor_columns[self.__attribute] == value]
            examples_for_child_target_col = self.__target_column[self.__predictor_columns[
                                                                     self.__attribute] == value]  # target values corresponding to the subset of examples with the given value

            # we grabbed the values from the examples themselves, so there should
            # be at least one example that has each value, but just in case there isn't
            # I don't want to crash the program
            if examples_for_child_target_col.empty:
                print("error: we shouldn't get here")

            # there are no columns left to use for decisions at the child
            # so lets make a leage node based on the most common target value in those examples
            elif len(examples_for_child_predictor_cols.columns.values) == 1:
                # create a child with the most common target value here
                leaf_child = DLeaf(self.__most_common_value_here)
                self.__child_nodes[value] = leaf_child

            # if all child examples have the same target value, we make a leaf node
            elif len(examples_for_child_target_col.unique()) == 1:  # all child examples have same class
                leaf_child = DLeaf(examples_for_child_target_col.unique()[0])  # make leaf with that class
                self.__child_nodes[value] = leaf_child  # put the leaf in the dictionary of children nodes

            else:  # we have a regular decision node for this attribute value
                # get rid of the column for this attribute so it can't be selected again
                examples_for_child_predictor_cols = examples_for_child_predictor_cols.drop(self.__attribute, 1)

                new_child = DNode(examples_for_child_predictor_cols, examples_for_child_target_col)
                new_child.train()  # generate the rest of the subtree for this child
                self.__child_nodes[value] = new_child  # put the new child node in the dictionary of children nodes

    # print out the tree - not the prettiest, but you can see it.
    def print_node(self, num_indents=0):
        for i in range(num_indents):
            print(" ", end=''),  # print with no newline
        print(self.__attribute)
        for attr in self.__child_nodes.keys():
            for i in range(num_indents):
                print("|", end='')
            print(":" + attr)
            self.__child_nodes[attr].print_node(num_indents + 1)

    # make a prediction for a single new example
    # this only makes sense to call after the tree has been build (with train())
    def predict(self, new_example):
        # look up the right branch in our dictionary of children
        if new_example[self.__attribute] in self.__child_nodes:
            node_on_corresponding_branch = self.__child_nodes[new_example[self.__attribute]]
            return node_on_corresponding_branch.predict(new_example)  # recursively call predict on the child node
        else:
            return self.__most_common_value_here  # there was no child, so we predict the most common class of the examples at this node


# class for representing a leaf node in the tree
class DLeaf:
    # when we create the node, all we need to know is what we're going to predict if we get here
    def __init__(self, val_in_target_col):
        self.__target_value = val_in_target_col

    # just returns the prediction for a new example,
    # this was probably called from predict() of a regular node one level up in the tree
    def predict(self, new_example):
        return self.__target_value

    # for displaying the tree
    def print_node(self, num_indents=0):
        for i in range(num_indents):
            print(" ", end='')
        print("LEAF:", self.__target_value)


# simply compares two Pandas series and returns the proportion that match
# this can be used to compute the accuracy of the prediction list against
# the actual target column
def accuracy(series1, series2):
    correct = 0.0
    for index, value in series1.iteritems():
        if value == series2.loc[index]:
            correct += 1
    return (correct / len(series1))

credit_data = pandas.read_csv('german.csv')

train_data, test_data = cross_validation.train_test_split(credit_data,test_size=0.1)

attributes_to_use = ['Status of existing checking account' , 'Credit history', 'Purpose']

my_tree = DTree()

my_tree.fit(train_data[attributes_to_use],train_data['Creditability'])
#my_tree.print_tree()
predictions = my_tree.predict(test_data[attributes_to_use])
#print(accuracy(test_data['Creditability'], predictions))
#getEntropy(credit_data)


entropyWHOLE = entropy(credit_data)
expectedEntropy = expectedEntropy(credit_data['Status of existing checking account'])
informationGain = entropyWHOLE - expectedEntropy
print(entropyWHOLE)
print(expectedEntropy) #this is getting wrong numbers because it's not weighted by instances down that branch
print(informationGain)
print(credit_data['Status of existing checking account'].value_counts())


#print(len(credit_data.loc[credit_data['Creditability']== 1]))
#col = credit_data['Creditability']
#print(col.value_counts().ix[1])
#print(test.loc[2])
#print(credit_data['Creditability'])
#male_trips.groupby('start_station_id').size()
