from __future__ import division
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


# Reading from files :
f = open ( 'data.txt' , 'r')
x = [[float(num) for num in line.split(',')] for line in f ]
f2 = open ( 'label.txt' , 'r')
y = [[int(num) for num in line.split(',')] for line in f2 ]


k = input("Enter the number of clusters (K) : ")
n = len(y)
m = len(y[0])
X = np.array(x)

def K_Means(k):

	# K Means : (SKlearn implementation)
	kmeans = KMeans(n_clusters=k, random_state=None).fit(X)


	# labelling i'th datapoint in matrix :
	labelled_mat = np.zeros((n,1))
	for i in range(0,n):
		for j in range(0,m):
			if(y[i][j] == 1):
				labelled_mat[i,0] = int((j+1) % 10)
				break
		

	mat = np.zeros((k,10))
	for i in range(0,n):
		mat[kmeans.labels_[i],int(labelled_mat[i,0])] = mat[kmeans.labels_[i],int(labelled_mat[i,0])] + 1

	R = len(mat)
	C = len(mat[0])	


	# cluster vector containing cluster labels :
	cluster_vector = np.zeros((1,R))

	for i in range(0,R):
		temp = 0
		for j in range(0,C):
			if(mat[i,j] > temp):
				temp = mat[i,j]
				cluster_vector[0,i] = j % 10


	# filling confusion matrix : 
	confusion_mat = np.zeros((10,10))
	for i in range(0,k):
		for j in range(0,10):
			confusion_mat[int(cluster_vector[0,i]),j] = confusion_mat[int(cluster_vector[0,i]),j] + mat[i,j]


	# summing rows :
	Summing_rows = np.sum(mat, axis=1)
	for i in range(0,R):
		Summing_rows[i] = Summing_rows[i] - mat[i,int(cluster_vector[0,i])]


	# finding accuracy :
	wrong_classified = np.sum(Summing_rows)
	accuracy = ((n-wrong_classified)/n)*100
	return (accuracy,confusion_mat)


# print output :
accuracy, confusion_mat = K_Means(k)
print('\n =========> Classification Accuracy is : '+ str(accuracy)+' %\n')


# plot drawn by varying k :
print('Drawing Plot for Accuracy vs number of clusters : ')
error_rates = []
for k in range(1,26):
    temp,cmat = K_Means(k)
    error_rates.append(temp)
    print(str(k)+' ===> '+ str(temp)+' %')


# Show the error rates
x =[]
for i in range(0,25):
	x.append(i+1)

y = error_rates
plt.plot(x,y,color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
plt.title('Accuracy vs number of clusters ')
plt.ylabel('Accuracy in %')
plt.xlabel('Number of clusters')
plt.legend() 
plt.savefig('Acc_Vs_clusters.png')










