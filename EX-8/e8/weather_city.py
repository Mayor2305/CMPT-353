import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from skimage.color import rgb2lab, lab2rgb, rgb2hsv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings("ignore")

labeled_file = sys.argv[1]
unlabeled_file = sys.argv[2]
output_file = sys.argv[3]

labled_data = pd.read_csv(labeled_file)
unlabled_data = pd.read_csv(unlabeled_file)

X = labled_data[['tmax-01','tmax-02','tmax-03','tmax-04','tmax-05','tmax-06','tmax-07','tmax-08','tmax-09','tmax-10','tmax-11','tmax-12','tmin-01','tmin-02','tmin-03','tmin-04','tmin-05','tmin-06','tmin-07','tmin-08','tmin-09','tmin-10','tmin-11','tmin-12','prcp-01','prcp-02','prcp-03','prcp-04','prcp-05','prcp-06','prcp-07','prcp-08','prcp-09','prcp-10','prcp-11','prcp-12','snow-01','snow-02','snow-03','snow-04','snow-05','snow-06','snow-07','snow-08','snow-09','snow-10','snow-11','snow-12','snwd-01','snwd-02','snwd-03','snwd-04','snwd-05','snwd-06','snwd-07','snwd-08','snwd-09','snwd-10','snwd-11','snwd-12']]
y = labled_data[['city']]

# classifier = VotingClassifier([
#     ('nb', GaussianNB()),
#     ('knn', KNeighborsClassifier(5)),
#     ('tree1', DecisionTreeClassifier(max_depth=4)),
#     ('tree2', DecisionTreeClassifier(min_samples_leaf=10)),
#     ('rf', RandomForestClassifier(n_estimators=100)),
#     ('SVC', SVC(kernel='linear', C=2.0)),
#     ('Svc', SVC())
# ])

model = make_pipeline(
    StandardScaler(),
    # classifier
    # RandomForestClassifier(n_estimators=100)
    SVC(kernel='linear', C=2.0)
    # GradientBoostingClassifier()
)

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model.fit(X_train, y_train)
print(model.score(X_valid, y_valid))

X_unlabled = unlabled_data[['tmax-01','tmax-02','tmax-03','tmax-04','tmax-05','tmax-06','tmax-07','tmax-08','tmax-09','tmax-10','tmax-11','tmax-12','tmin-01','tmin-02','tmin-03','tmin-04','tmin-05','tmin-06','tmin-07','tmin-08','tmin-09','tmin-10','tmin-11','tmin-12','prcp-01','prcp-02','prcp-03','prcp-04','prcp-05','prcp-06','prcp-07','prcp-08','prcp-09','prcp-10','prcp-11','prcp-12','snow-01','snow-02','snow-03','snow-04','snow-05','snow-06','snow-07','snow-08','snow-09','snow-10','snow-11','snow-12','snwd-01','snwd-02','snwd-03','snwd-04','snwd-05','snwd-06','snwd-07','snwd-08','snwd-09','snwd-10','snwd-11','snwd-12']]

predictions = model.predict(X_unlabled)

pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)