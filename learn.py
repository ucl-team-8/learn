import csv
from sklearn import tree

def get_features(props):
    return [
        int(props['total_matching']),
        float(props['matched_over_total']),
        float(props['median_time_error']),
        float(props['iqr_time_error'])
    ]

def process_file(filename):
    the_file = open(filename, 'r')
    reader = csv.DictReader(the_file)
    return [get_features(x) for x in reader]

planned = process_file('planned_service_matchings_northern_rail.csv') + process_file('planned_service_matchings_east_coast.csv')
unplanned = process_file('unplanned_service_matchings_northern_rail.csv') + process_file('unplanned_service_matchings_east_coast.csv')

# remove the ones with very few matched events
planned = filter(lambda x: x[0] > 2, planned)

X = planned + unplanned
Y = [1] * len(planned) + [0] * len(unplanned)

clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X, Y)

# export dot
from sklearn.externals.six import StringIO
with open("matchings.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

# PDF can be generated using dot on command line:
# $ dot -Tpdf matchings.dot -o matchings.pdf

# persist learned model to disk
import pickle
pickle.dump(clf, open('decision_tree','w'))
