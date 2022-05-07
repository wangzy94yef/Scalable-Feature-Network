import pandas
from sklearn.model_selection import train_test_split
import numpy

seed = 7
cms_dataset = pandas.read_csv("./multiscale_dataset/cms/PU_dataset_1.csv", header=None)
# print(cms_dataset.head())
dataset = cms_dataset.values
line_cms = dataset[:, :10]
func_cms = dataset[:, 10:21]
lable = dataset[:, 21:]
# print(line_cms)
# print(func_cms)
# print(lable)


line_cms_train, line_cms_test, func_cms_train, func_cms_test, lable_train, lable_test \
    = train_test_split(line_cms, func_cms, lable, test_size=0.25, random_state=seed)

# print(func_cms_train)
# print(line_cms_test)
# print(lable_train)
# print(lable_test)