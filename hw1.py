import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from KNN import KNN

iris = load_iris()
X = iris.data[:, :4] # for now, use the first two features.
y = iris.target
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5 # 0.5씩 여유두는것
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(figsize=(8, 6))
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,#colormap>> 색 지정
 edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal widtCLEANh')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
#plt.show()

KNN.get_distance(X,y)  # knn 클래스 실행
KNN.get_neighbors()



