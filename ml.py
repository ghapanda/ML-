import copy

import math
import numpy as np
import time
import matplotlib.pyplot as plt
from tabulate import tabulate
# def f(x):
#     return  np.sin(1-np.cos(x)) / (1-np.cos(x))
#
#
# # x=np.linspace(-5,5,3000)
# # y=np.sin(1-np.cos(x)) / (1-np.cos(x))
# # plt.plot(x,y)
# # plt.show()
# pos_x=[10**(-i) for i in range(5)]
# neg_x=[-i for i in pos_x]
# f_pos=np.vstack([f(i) for i in pos_x])
# f_neg=np.vstack([f(i) for i in neg_x])
# pos_x=np.vstack((pos_x))
# neg_x=np.vstack((neg_x))
# x_values=np.hstack(((pos_x, f_pos, neg_x, f_neg)))
# table=tabulate(x_values, headers= ["pos x","pos f(x)", "neg x", "neg f(x)"], tablefmt="github")
# print(table)
# import numpy as np
# x_train=np.array(list(map(int,input('enter the feature variable of the training data set: ').split())))
# y_train=np.array(list(map(int,input('enter the target variable for the training data set: ').split())))
# print(f' x_train = {x_train}')
# print(f'y_train = {y_train}')
#
# def compute_model_output(x,w,b):
#     m=x.shape[0]
#     f_wb= np.zeros(m)
#     for i in range(m):
#         f_wb[i]=w*x[i]+b
#     return f_wb
# f_vls=compute_model_output(x_train,200,200)
# plt.scatter(x_train,y_train,marker="*",c='r')
# plt.plot(x_train,f_vls,c='b')
# plt.title("our model for price prediction ")
# plt.xlabel("size in sqrft")
# plt.ylabel("price in 1000 dollars")
# plt.show()
# import numpy as np
# x_train=np.array(list(map(float,input("enter feature variable of the training set: ").split(","))))
# y_train=np.array(list(map(float,input("enter target variable of the training set:").split(","))))
#
#
# plt.scatter(x_train,y_train,marker="*",c='b')
#
# m=x_train.shape[0]
# fw_b=np.zeros(m)
# def model(x,w,b):
#
#     for i in range(m):
#         fw_b[i]=w*x_train[i]+b
#     return fw_b
# w=float(input("enter w:"))
# b=float(input("enter b: "))
#
# result=model(x_train,w,b)
# plt.plot(x_train,result,c='r')
# plt.title("y^ vs y")
# plt.xlabel("feature var")
# plt.ylabel("target var")
# plt.show()
# x_train=np.vstack((x_train))
# y_train=np.vstack((y_train))
# y=np.vstack((fw_b))
# values=np.hstack((x_train,y_train,y))
# table=tabulate(values,headers=["feature","target","model"])
# print(table)
# import time
#
# x = np.array([[[0, 1, 2], [0, 1, 2]], [[0, 3, 4], [0, 3, 4]]])
# print(x.shape)
# print(x.dtype)
# print(x.ndim)
# print(f'x= {x} is an array with shape {x.shape} and number of dimentions: {x.ndim}')
# a = np.linspace(0, 10, 6)
# b = np.linspace(10, 20, 6)
# print(a)
# print(b)
#
#
# def dot_product(p, q):
#     try:
#         Dot = 0
#         for i, j in zip(p, q):
#             Dot += i * j
#         return Dot
#
#
#
#
#     except Exception as e:
#         print("the error message is :")
#         print(e)
#
#
# start = time.time()
# print(dot_product(a, b))
# end = time.time()
# d = end - start
#
# print("duration is {}".format(d))
# start = time.time()
# print(np.dot(a, b))
# end = time.time()
# d = end - start
# print("duration is {}".format(d))


x_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
def standard(x):
    m=x.shape[0]
    mean=sum(x)/m
    sum2=0
    for i in range(m):
        sum2+=(x[i]-mean)**2
    return math.sqrt(sum2/m)



def normalize(x):
    m=x.shape[0]
    n=x.shape[1]
    sum=0
    for i in range(m):
        for j in range(n):
            sum+=x[i,j]
        mean=sum/n
        for j in range(n):
            x[i,j]=(x[i,j]-mean)/(standard(x[i]))

    return x
x_train=normalize(x_train)
print(x_train)


y_train=np.array([460,232,178])
print(f"x_train has {x_train.shape[0]} training examples and {x_train.shape[1]} features")
print(f"y_train is {y_train}")

w_init=np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
b_init = 785.1811367994083
print(f" the initial value of w is {w_init} and for b is {b_init}")

def compute_cost(x,y,w,b):
    m=x.shape[0]
    sumOfSqu=0
    for i in range(m):
        sumOfSqu+=(np.dot(x[i],w)+b-y[i])**2

    sumOfSqu/=2*m
    return sumOfSqu
#
# def compute_gradient(x,y,w,b):
#     m,n=x.shape
#     dj_dw=0
#     dj_db=0
#     for i in range(m):
#         error=np.dot(x[i],w)+b-y[i]
#         for j in range(n):
#             dj_dw+=error*x[i,j]
#         dj_db+=error
#     dj_db/=m
#     dj_dw/=m
#     return dj_dw, dj_db
# def gradient_descent(x,y,w_init,b_init,compute_cost, compute_gradient,
#                                                     alpha, iter):
#     w=copy.deepcopy(w_init)
#     b=b_init
#     j_his=[]
#     for i in range(iter):
#         dj_dw,dj_db=compute_gradient(x,y,w,b)
#         w-=alpha*dj_dw
#         b-=alpha*dj_db
#         j=compute_cost(x,y,w,b)
#         j_his.append(j)
#         if (i%100==0):
#             print(f"the cost for this w={w} and b={b} is {j}")
#     return w,b,j_his
# initial_w = np.zeros_like(w_init)
# initial_b = 0.
# # some gradient descent settings
# iterations = 1000
# alpha = 5.0e-7
# # run gradient descent
# w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
#                                                     compute_cost, compute_gradient,
#                                                     alpha, iterations)
# print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
# m,_ = x_train.shape
# for i in range(m):
#     print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
# Iteration=np.arange(iterations)
#
# plt.plot(100+np.arange(len(J_hist[100:])),J_hist[100:])
#
# plt.show()



def compute_gradient(x,y,w,b):
    m=x.shape[0]
    n=x.shape[1]
    dj_dw=0
    dj_db=0
    for i in range(m):
        error=np.dot(x[i],w)+b-y[i]
        for j in range(n):
            dj_dw+=error*x[i,j]
        dj_db+=error
    dj_db/=m
    dj_dw/=m
    return dj_dw,dj_db
def gradient_descent(x,y,w_init,b_init,alpha,iteration,compute_gradient,compute_cost):
    w=copy.deepcopy(w_init)
    b=b_init
    m=x.shape[0]
    dj_dw,dj_db=compute_gradient(x,y,w,b)
    j_his=[]
    for i in range(iteration):
        w-=alpha*dj_dw
        b-=alpha*dj_db
        j=compute_cost(x,y,w,b)

        j_his.append(j)
        if i%100000==0:
            print(f"for w= {w} and b= {b}, our lost is {j_his[-1]}")
    return w,b,j_his
w_in=np.zeros_like(w_init)
b_in=0
alpha=9.0e-15
iteration=int(10000000000)
w_final,b_final,j_his=gradient_descent(x_train,y_train,w_in,b_in,alpha,iteration,compute_gradient,compute_cost)
print(f"our final reslut: w={w_final},b={b_final},cost={j_his[-1]} with {iteration} number of iterations")
plt.plot(int(900)+np.arange(len(j_his[int(900):])),j_his[int(900):],c='r')
plt.show()


































