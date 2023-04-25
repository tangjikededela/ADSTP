from pandas import read_csv
import IntegratedPipeline as IP

# Set pipelines
pipeline = IP.general_datastory_pipeline
# Set replace variables names
readable_names = dict((kv.split(': ') for kv in (l.strip(' \n') for l in open('./data/readableNamesForTenData.txt'))))

# # 3. Fish
# fish_dataset = read_csv('./data/fish.csv')
# Xcol = ['Length', 'Diagonal', 'Height', 'Width']
# ycol = 'Weight'
# pipeline.LinearFit(fish_dataset, Xcol, ycol, [readable_names.get(key) for key in Xcol], readable_names.get(ycol))
#
# # 4. Insurance
# insurance_dataset = read_csv('./data/insurance.csv')
# Xcol = ['age', 'bmi', 'children']
# ycol = 'charges'
# pipeline.LinearFit(insurance_dataset, Xcol, ycol, [readable_names.get(key) for key in Xcol], readable_names.get(ycol))
#
# # 6. Cancer (the dataset need to transform into utf-8 to work)
# cancer_dataset = read_csv('./data/cancer_reg.csv')
# Xcol = ['incidenceRate', 'medIncome', 'popEst2015', 'povertyPercent', 'MedianAge']
# ycol = 'TARGET_deathRate'
# pipeline.LinearFit(cancer_dataset, Xcol, ycol, [readable_names.get(key) for key in Xcol], readable_names.get(ycol))

# 7. Estate
estate_dataset = read_csv('./data/Real estate.csv')
Xcol = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
        'X4 number of convenience stores']
ycol = 'Y house price of unit area'
pipeline.LinearFit(estate_dataset, Xcol, ycol, [readable_names.get(key) for key in Xcol], readable_names.get(ycol))

# 8. Red wine
# There is already an example in Example 2_1

# # 9. Car
#
# Car_dataset = read_csv('./data/car data.csv')
#
# Xcol = ['Present_Price', 'Kms_Driven', 'Year']
# ycol = 'Selling_Price'
#
# # fit the model
# pipeline.LinearFit(Car_dataset, Xcol, ycol, [readable_names.get(key) for key in Xcol], readable_names.get(ycol))
