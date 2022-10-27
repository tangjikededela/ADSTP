from pandas import read_csv
import controller as CT

# Read the example dataset about crime rate
features = read_csv('./data/attributes.csv', delim_whitespace=True)
dataset = read_csv('./data/communities.data', names=features['attributes'])
dataset = dataset.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], axis=1)
dataset = dataset.drop(columns=['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1)
# Data cleaning
dataset = CT.cleanData(dataset, 0.8)  # Clear data with a threshold of 80%
# Read the more readable variable names
readable_names = dict((kv.split(': ') for kv in (l.strip(' \n') for l in open('./data/readableNames.txt'))))
# Generate the data stories on dashboard
CT.LinearModelStats(dataset, ['pctWPubAsst','PctHousLess3BR','PctPersOwnOccup'], 'ViolentCrimesPerPop', [readable_names.get(key) for key in ['pctWPubAsst','PctHousLess3BR','PctPersOwnOccup']],readable_names.get('ViolentCrimesPerPop'), questionset=[1, 1, 1, 1], trend=1)