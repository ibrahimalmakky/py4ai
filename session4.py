from session3_part2 import load_split_iris
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

num_runs = 1
sum_svm_acc = 0
sum_kmeans_acc = 0

for i in range(0, num_runs):

    x_train, x_test, y_train, y_test = load_split_iris()

    svm = SVC()
    svm.fit(x_train, y_train)

    svm_pred = svm.predict(x_test)

    svm_acc = metrics.accuracy_score(y_test, svm_pred)
    sum_svm_acc += svm_acc

    kmeans = KMeans(n_clusters=3, tol=1000, max_iter=10000)

    kmeans.fit(x_train)

    kmeans_pred = kmeans.predict(x_test)
    print(kmeans_pred)
    print(y_test)

    my_plot = plt.subplot()

    cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
    my_plot.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=cm_bright)
    plt.show()

    plot_confusion_matrix(svm, x_test, y_test)
    plt.show()

    kmeans_acc = metrics.accuracy_score(y_test, kmeans_pred)
    
    sum_kmeans_acc += kmeans_acc


avrg_svm_acc = sum_svm_acc/num_runs
avrg_kmeans_acc = sum_kmeans_acc/num_runs

print("SVM: "+ str(avrg_svm_acc))
print("Kmeans: " + str(avrg_kmeans_acc))
