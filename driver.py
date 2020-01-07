from DecisionTree import *
import pandas as pd
from sklearn import model_selection

header = ['Name','Hobby','Age','EducationalLevel','MStatus','Class']

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.data', header=None, names=['Name','Hobby','Age','EducationalLevel','MStatus','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
t_pruned = prune_tree(t, [283,70,138])

print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t_pruned)
print("Accuracy on test = " + str(acc))




