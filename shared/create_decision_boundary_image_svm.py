import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib
from sklearn.preprocessing import StandardScaler


# Creates random sample data and fits a SVM to show the hyperplane, margin and support vectors
# Modified from https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html

def createDecisionBoundaryImageSVM():
    matplotlib.rcParams.update({'font.size': 16})
    
    x1 = np.random.rand(30,2)+0.5
    x2 = np.random.rand(30,2)
    y1 = np.ones(30)
    y2 = np.zeros(30)
    X = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')

    plt.savefig("./svm_decision_boundary.png", dpi=1000, bbox_inches='tight')