from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def load_split_iris():
    iris = datasets.load_iris()
    # print(iris.data.shape)
    # print(iris.target.shape)
    inputs = iris.data
    targets = iris.target
    # inputs = inputs[:,0:2]
    # print(inputs.shape)
    # print(targets.shape)
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)
    return x_train, x_test, y_train, y_test 

def main():
    x_train, x_test, y_train, y_test = load_split_iris()

    random_forest = RandomForestClassifier(n_estimators=100)

    random_forest.fit(x_train, y_train)

    pred = random_forest.predict(x_test)
    print(y_test)
    print(pred)

    acc = metrics.accuracy_score(y_test, pred)
    print(acc)

    conf_matrix = metrics.confusion_matrix(y_test, pred)
    print(conf_matrix)

    print(random_forest.feature_importances_)
    # print(iris.feature_names)

if __name__ == "__main__":
    main()
