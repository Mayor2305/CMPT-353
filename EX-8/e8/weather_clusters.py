import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_pca(X):
    """
    Transform data to 2D points for plotting. Should return an array with shape (n, 2).
    """
    flatten_model = make_pipeline(
        # StandardScaler(),
        MinMaxScaler(),
        PCA(2, svd_solver='full')
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2


def get_clusters(X):
    """
    Find clusters of the weather data.
    """
    model = make_pipeline(
        # StandardScaler(),
        # MinMaxScaler(),
        KMeans(n_clusters=10)
    )
    model.fit(X)
    return model.predict(X)


def main():
    data = pd.read_csv(sys.argv[1])

    X = data[['year','tmax-01','tmax-02','tmax-03','tmax-04','tmax-05','tmax-06','tmax-07','tmax-08','tmax-09','tmax-10','tmax-11','tmax-12','tmin-01','tmin-02','tmin-03','tmin-04','tmin-05','tmin-06','tmin-07','tmin-08','tmin-09','tmin-10','tmin-11','tmin-12','prcp-01','prcp-02','prcp-03','prcp-04','prcp-05','prcp-06','prcp-07','prcp-08','prcp-09','prcp-10','prcp-11','prcp-12','snow-01','snow-02','snow-03','snow-04','snow-05','snow-06','snow-07','snow-08','snow-09','snow-10','snow-11','snow-12','snwd-01','snwd-02','snwd-03','snwd-04','snwd-05','snwd-06','snwd-07','snwd-08','snwd-09','snwd-10','snwd-11','snwd-12']]
    y = data['city']
    
    X2 = get_pca(X)
    clusters = get_clusters(X)
    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=20)
    plt.savefig('clusters.png')
    
    df = pd.DataFrame({
        'cluster': clusters,
        'city': y,
    })
    counts = pd.crosstab(df['city'], df['cluster'])
    print(counts)


if __name__ == '__main__':
    main()
