
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1 - Data pre-processing

# Read in CSV as pandas data frame
data = pd.read_csv("data.csv")

# Clear out unnamed columns from the data
unnamed_columns = data.columns.str.contains('unnamed', case=False)
data = data.drop(data.columns[unnamed_columns], axis=1)

# Get rid of the data index column
data = data.drop("id", axis=1)


# 2 - Exploratory analysis

# The dataset contains the means, standard errors, and average of the 3 worst
# values; the means are in columns 1:11, ses in 11:21, and worsts in 21:31
means = data[data.columns[1:11]]
ses = data[data.columns[11:21]]
worsts = data[data.columns[21:31]]
print(data.shape)

# Plot the correlations between the worsts
# sns.pairplot(data, hue="diagnosis", vars=data.columns[1:11])
# plt.tight_layout()
# plt.savefig("correlations.svg")
# plt.close()

# Calculate the correlation matrix between the means and plot the heatmap
# corr = means.corr()
# axes = sns.heatmap(corr, annot=True, square=True, fmt=".2f")
# plt.tight_layout()
# plt.savefig("corr matrix.svg")
# plt.close()


# 3 - Exploratory PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardise all the data
x = data[data.columns[1:31]].values
y = data["diagnosis"].values
x = StandardScaler().fit_transform(x)

# Perform PCA
# pca = PCA()
# components = pca.fit_transform(x)

# Get the data relative to the principle components
# pc_data = pd.DataFrame(data=components)

# Graph how much of the variance in the original data is explained by each
# principle axis
# sns.barplot(np.arange(0, 30), pca.explained_variance_ratio_ * 100)
# plt.title("Variance Explained by Principle Components")
# plt.xlabel("Principle Component Number")
# plt.ylabel("% Variance Explained by Component")
# plt.tight_layout()
# plt.show()

# Find how many principle components are needed to explain 85% of the variance
# in the original data
# total = 0
# count = 0
# while total < 0.85:
# 	total += pca.explained_variance_ratio_[count]
# 	count += 1
# print(count, total)

# 6 principle components explain 88.76% of the variance in our original data, so
# use the first 6 principle components only
n_components = 6


# 3 - PCA

# Perform PCA, selecting the top 6 components
pca = PCA(n_components=n_components)
components = pca.fit_transform(x)

# Get the data relative to the principle components
pc_data = pd.DataFrame(data=components)


# 4 - Train SVM model

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import sklearn.metrics

# Replace the malignant/benign column with 0 = benign, and 1 = malignant
diagnosis = data.diagnosis.apply(lambda x: 1 if x == "M" else 0)

# Split the data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(pc_data, diagnosis, 
	test_size=0.3, random_state=42)

# Ensure the class distribution in the testing and training sets are roughly
# equal
# print("Train proportions: (benign, malignant)", 
# 	pd.value_counts(y_train).values.astype(float) / float(y_train.size))
# print("Test proportions: (benign, malignant)",
# 	pd.value_counts(y_test).values.astype(float) / float(y_test.size))

# Train a linear SVM
svm = LinearSVC(random_state=42)
svm.fit(X_train, y_train)

# Measure accuracy
predicted = svm.predict(X_test)
print("Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, predicted)))

# Measure precision and recall
# print("Precision: " + str(sklearn.metrics.precision_score(y_test, predicted)))
# print("Recall: " + str(sklearn.metrics.recall_score(y_test, predicted)))
report = sklearn.metrics.classification_report(y_test, predicted, digits=4)
print(report)
