import pandas as pd;
import numpy as np;
import os;
import sys;
from scipy.fftpack import fft, dct
from skimage.feature import hog
from skimage.filters import gabor

def normalise(X):
	mean = np.sum(X)/(X.size)
	# print(mean)
	var = np.sum((X-np.full((X.shape),mean))**2)/(X.size)
	# print(var)
	# print(X[0][:10])
	# print(((X-mean)/var)[0][:10])
	return (X-mean)/var, mean, var

def addFeatures(X):
	fftFeatures = np.array(np.abs(fft(X,axis=1)))
# 	dctFeatures = np.array(np.abs(dct(X,type=2,axis=1,norm='ortho')))
	hogFeatures = np.full((X.shape[0],32),0)
# 	gaborFeatures = np.full((X.shape[0],1024),0)
	for o in range(X.shape[0]):
		squareImage = np.reshape(X[o,:],(32,32))
		temp1 = hog(squareImage,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1))
# 		temp2_r, temp2_i = gabor(squareImage,fequency=0.6)
		hogFeatures[o,:] = temp1.flatten()
# 		gaborFeatures[o,:] = temp2_r.reshape(1,1024);

# 	X = np.concatenate((X,dctFeatures/500),axis=1)
	X = np.concatenate((X,fftFeatures/500),axis=1)
	X = np.concatenate((X,hogFeatures*500),axis=1)
# 	X = X[1024:,:].copy()
	return X

def oneHot(y):
	y = np.array(y)
	B = np.zeros((y.shape[0],10))
	for i in range(y.shape[0]):
		for j in range (10):
			t = y[i]
			B[i,t] = 1
	return B

def h(Z): # softmax function
	Z = np.exp(Z)
	for i in range(Z.shape[0]):
		sum1 = np.sum(Z[i,:])
		Z[i,:] = Z[i,:]/sum1
	return Z

def cee(y,ycap,size):

	return (1/size)*np.sum(y*np.log(ycap))

def f(str,a):
        if str == 'softplus':
            return np.log(1+np.exp(a))
        if str == 'sigmoid':
            return 1/(1+np.exp(-a))
        if str == 'tanh':
            return np.tanh(a)
        if str == 'relu':
            return np.maximum(a,0)
        if str == 'pararelu':
            a = np.array(a)
            return np.array((a>0)*a+(a<0)*0.01*a)
        if str == 'linear':
        	return a;
        else:
    	    print("did not recognise activation function")

def fdash(str,a):
    if str == 'pararelu':
        a = np.array(a)
        return (a>0)*1+(a<0)*0.01
    if str == 'relu':
        return (a>0)*1
    b = f(str,a)
    if str == 'softplus':
            return f('sigmoid',a)
    if str == 'sigmoid':
        return b*(1-b)
    if str == 'tanh':
        return 1-b**2
    if str == 'linear':
    	return 1;

def forwardProp(w,A,Z,l,b,act):
	for k in range(1,l+1):
		Z[k] = A[k-1]@w[k]+b[k]
		A[k] = f(act[k-1],Z[k])
	Z[l+1] = A[l]@w[l+1]+b[l+1]
	A[l+1] = h(Z[l+1])
	return A, Z, w, b

def backProp(i,w,A,Z,y,b,ycap,l,alpha,delLdelw,delLdelb,act):
	size = y.shape[0]
	delLdelz = (1/size)*(y-ycap)
	for k in range(l+1,0,-1):
		delLdelw[k] = A[k-1].T@delLdelz
		delLdelb[k] = np.sum(delLdelz,0)
		delLdela = delLdelz@w[k].T
		if k > 1:
			delLdelz = delLdela*(fdash(act[k-2],Z[k-1]))
	return delLdelw, delLdelb

def update(w,b,delLdelw,delLdelb,l,a,size,i,j):
	for k in range(1,l+2):
		b[k] = b[k] + a*delLdelb[k]
		w[k] = w[k] + a*delLdelw[k]
	return w, b, i+1

def init(k,X,Y,w,b,wmul,bmul):
	last = X.shape[1]
	k = np.array(k)
	l = k.shape[0]

	for i in range(1,l+1):
		curr = k[i-1]                                              
		w[i] = np.random.uniform(-1,1,(last,curr))*wmul
		b[i] = np.random.uniform(-1,1,(1,curr))*bmul
		last = curr
	w[l+1] = np.random.uniform(-1,1,(last,Y.shape[1]))*wmul
	b[l+1] = np.random.uniform(-1,1,(Y.shape[1]))*bmul
	return w, b, l

def neural_d(train,test,op):
	train = pd.read_csv(train,header=None).values
	np.random.shuffle(train)
	Xtr = train[:18000,:train.shape[1]-1].copy()
	Xtr = addFeatures(Xtr);
	print(Xtr.shape)
	# Xtr, mean, var = normalise(Xtr)
	Ytr = train[:18000,train.shape[1]-1].copy()
	Ytr = Ytr.reshape(Ytr.shape[0],1)
	Ytr_notEn = Ytr
	Ytr = oneHot(Ytr)

	[m,n] = Xtr.shape

	## READING PARAMETERS
	mode = 2
	alpha = 0.15
	iterations = 2000
	size = 500
	k = [512,512,256]
	l = len(k)
	sub_iters = m//size
	w = {}
	b = {}
	w, b, l = init (k,Xtr,Ytr,w,b,wmul=0.01,bmul=1)
	act = ['pararelu','pararelu','tanh']


	i = 0
	while i<iterations:
		for j in range(sub_iters):
			if i>= iterations:
				break;
			x = Xtr[j*size:(j+1)*size,:]
			y = np.zeros((size))
			y = Ytr[j*size:(j+1)*size,:]

			A = {}
			Z = {}
			A[0] = x
			Z[0] = x
			A, Z, w, b = forwardProp(w,A,Z,l,b,act)

			a = alpha
			if mode == 2:
				a = alpha/np.sqrt(i+1)		
			
			delLdelw = {}
			delLdelb = {}

			delLdelw, delLdelb = backProp(i,w,A,Z,y,b,A[l+1],l,a,delLdelw,delLdelb,act)
			w, b, i = update(w,b,delLdelw,delLdelb,l,a,size,i,j)
			print(i)
# 		Att = {}
# 		Att[0] = Xtr
# 		Ztt = {}
# 		Ztt[0] = Xtr
# 		Att,Ztt,w,b = forwardProp(w,Att,Ztt,l,b,act)
# 		pred = np.argmax(Att[l+1],1)
# 		pred = pred.reshape(Ytr_notEn.shape[0],1)
# 		tot = 0
# 		for u in range(Ytr_notEn.shape[0]):
# 		    if Ytr_notEn[u] == pred[u]:
# 		        tot = tot + 1
		print(i//1,'/',iterations//1,w[l+1][:10])
	test = pd.read_csv(test,header=None).values
	Xt = test[:,:test.shape[1]-1].copy()
	Xt = addFeatures(Xt)
	At = {}
	Zt = {}
	At[0] = Xt
	Zt[0] = Xt
	At, Zt, w, b = forwardProp(w,At,Zt,l,b,act)
	
	pred = np.argmax(At[l+1],axis=1)
	pred = pred.reshape(Xt.shape[0],1)
	print('pred',pred[:10])
	print(pred.shape)
	for i in pred:
	    print(np.asscalar(i),file=open(op,"a"))


if __name__ == '__main__':
	neural_d(*sys.argv[1:])