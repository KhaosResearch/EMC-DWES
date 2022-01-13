import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import tree,svm
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage,fcluster
import random
import math
from sklearn.metrics import confusion_matrix
from time import time
# from sklearn.metrics.cluster import normalized_mutual_info_score


# colon; labels are 0 or 1
# df = pd.read_excel ('...\colon.xlsx', header=None)
# df.iloc[:,df.shape[1]-1].replace({'Normal':1, 'Tumor':2},inplace=True)

# CNS; labels are 1 or 2
# df = pd.read_excel ('...\CNS.xlsx', header=None)
# df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# Leukemia-2c; labels are 1 or 2
# df = pd.read_excel ('...\Leukemia.xlsx', header=None)
#
#SMK
# df = pd.read_csv ('...\SMK.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# GLI
# df = pd.read_csv ('...\GLI.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# Covid-2c
# df = pd.read_csv ('...\Covid.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({'no virus':1, 'other virus':1, 'SC2':2},inplace=True)

# Covid-3c
# df = pd.read_csv ('...\Covid.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({'no virus':1, 'other virus':2, 'SC2':3},inplace=True)


#Leukemia-3c
# df = pd.read_excel ('...\Leukemia_3c.xlsx', header=None)

#MLL-3c
# df = pd.read_excel ('...\MLL.xlsx', header=None)

#SRBCT-4c
# df = pd.read_excel ('...\SRBCT.xlsx', header=None)

# =========== Separate predict variables from response variable======================
X=df.iloc[:,0:df.shape[1]-1]
X=pd.DataFrame(scale(X))

y=df.iloc[:,df.shape[1]-1]

# ============= Just active lines 62 & 63 to use Mutual Congestion ===============

# ones=sum(df[df.shape[1]-1]==1)
# twos=sum(df[df.shape[1]-1]==2)

# ==============Run the following code and exchange X and y by Xn and yn in line 79 solely if the accuracy of EMC is investigated============================================

# Xn=mat1[:,0:mat1.shape[1]-1]
# Xn=pd.DataFrame(scale(Xn))
# yn=mat1[:,math.ceil(len(z)/RR)]
#============= Accuracy of dataset Without applying any Feature Selection method==================

#increasing iteration in for loop yeilds more accurate result

p_final=np.zeros(10)
r_final=np.zeros(10)
f_final=np.zeros(10)
s=np.zeros(10)
nm=np.zeros(10)
 

for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

  # dectree = tree.DecisionTreeClassifier()
  # dectree.fit(X_train,y_train)
  # s[i]=dectree.score(X_test,y_test)
  # nm[i]=normalized_mutual_info_score(dectree.predict(X_test),y_test)
  # cm=confusion_matrix(y_test,dectree.predict(X_test))
  svm_linear=svm.SVC(C=50,kernel="linear")
  svm_linear.fit(X_train,y_train)
  s[i]=svm_linear.score(X_test,y_test)

  cm=confusion_matrix(y_test,svm_linear.predict(X_test))
  p_final[i]= cm[1,1] / (cm[0,1]+cm[1,1])
  r_final[i]= cm[1,1] / (cm[1,1]+cm[1,0])
pre=np.mean(p_final) 
rec=np.mean(r_final) 
fscore=2 * ((pre*rec) / (pre+rec))
np.mean(s)
# np.mean(nm)

    
############# Mutual Congestion (MC) #########################
alpha=np.zeros(df.shape[1]-1)
sorted_alpha=np.zeros(df.shape[1]-1)
for i in range(df.shape[1]-1):
    print(i) 
    newdf=df.sort_values(i)
    # if labels start with 1, find the location of the first place!='1'
    if newdf.iloc[0,df.shape[1]-1]==1:
      first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(1).idxmax())
#      co=1
      co=first
      for j in range(first,newdf.shape[0]):
      
        if newdf.iloc[j,newdf.shape[1]-1]==1:
          co=co+1
        if co==ones:
          last=j
          break
      alpha[i]=(last-first)/(df.shape[0])
    else:
        first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(2).idxmax())
#        co=1
        co=first
        for j in range(first,newdf.shape[0]):
      
          if newdf.iloc[j,newdf.shape[1]-1]==2:
            co=co+1
          if co==twos:
            last=j
            break
        alpha[i]=(last-first)/(df.shape[0])
####################Extendec Mutual Congestion-multi label  (EMC) #############################
label=len(y.unique())
alpha=np.zeros(df.shape[1]-1)
sorted_alpha=np.zeros(df.shape[1]-1)
ar=np.zeros([df.shape[0],df.shape[1]-1])
for i in range(df.shape[1]-1):
    print(i)
    newdf=df.sort_values(i)
    ar[:,i]=newdf.iloc[:,df.shape[1]-1]
    cl=1
    R=np.zeros(label)
    C=np.zeros(label)
    arr=pd.DataFrame(ar)
    for lab in range(label):
           co=0
           first= arr[i].eq(cl).idxmax()+1
           last=arr.index.get_loc(arr.where(arr[i]==cl).last_valid_index())+1
            
           r1=0
           for j in range (first-1,last):
               if arr.iloc[j,i]==cl:
                   r1=r1+1
               else:
                   j1=j
                   break
           r2=0
           for j in range (last-1, first-1,-1):
               if arr.iloc[j,i]==cl:
                   r2=r2+1
               else:
                  j2=j 
                  break
              

           co=(j2-j1)+1
                 
           C[lab]=co
           R[lab]=r1+r2
           cl=cl+1
    alpha[i]=(np.sum(C))/(np.sum(C)+np.sum(R))
                    
               

####################construct MC df########################################  
# Retaining Rate of  EMC  (RR)=> 100=0.01 --> 20=0.05, by assigning RR=20 you indicate that 95% of the data are discarded
RR=20      
z=np.argsort(alpha)
mat1=np.zeros((df.shape[0],math.ceil(len(z)/RR)+1))
#mat1=np.zeros((df.shape[0],201))
for i in range(math.ceil(len(z)/RR)):
#
   mat1[:,i]=X.iloc[:,z[i]]
#mat1[:,200]=y
mat1[:,math.ceil(len(z)/RR)]=y

######################Hierarchical Clustering  MC###################

# cl_num is number of clusters in DWES

t0=time()
iter=200

accC=np.zeros(iter)


opt=0

Xn=mat1[:,0:mat1.shape[1]-1]

Xn=pd.DataFrame(scale(Xn))
yn=mat1[:,math.ceil(len(z)/RR)]


#Xn=X
#yn=y



# we want to cluster features to cl_num number of clusters 
# cl_num is the cluster numbers (q)
Xnt=Xn.T
cl_num=6
clusterer = linkage(Xnt, 'complete',metric='euclidean')
cluster_labels = fcluster(clusterer,cl_num,criterion='maxclust')  

cln=pd.DataFrame(cluster_labels)
cln.columns=['labels']

alfa=0.1

 
th=np.full((1, cl_num), 0.5)      # setup initial threshold

th1=np.zeros(cl_num) 

for i in range(cl_num):       # generate random threshold 
  th1[i]=random.uniform(0, 1)

mask=np.where(th1<=th)     # compare with initial threshold and select target clusters
mask=mask[1]+1   
  
subset_size=sum(th1<th)
 
# construct one solution
subset=np.zeros(len(mask))

for i in range(len(mask)):
    subset[i]=cln.labels[cln.labels.eq(mask[i])].sample().index.values

#construct the respective matrix
matn=np.zeros((df.shape[0],len(mask)+1))
# df_ = pd.DataFrame(index=61, columns=cl_num+1)
for i in range(len(mask)):
     matn[:,i]=Xn.iloc[:,int(subset[i])]
matn[:,len(mask)]=yn

# run SVM or DT for new mat
it=1
Xnew_n=matn[:,0:len(mask)]
ynew_n=matn[:,len(mask)]

s=np.zeros(10)
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(Xnew_n,ynew_n, test_size=0.2)
    svm_linear=svm.SVC(C=50,kernel="linear")
    svm_linear.fit(X_train,y_train)
    s[i]=svm_linear.score(X_test,y_test)
    # dectree = tree.DecisionTreeClassifier()
    # dectree.fit(X_train,y_train)
    # s[i]=dectree.score(X_test,y_test)
acc=np.mean(s)
s_size=np.zeros(20)
count=0;
s_size[count]=len(subset)  
print ("acc",  "  ", acc,"      ","subset_size", "  ", s_size[count],"     ", "iteration", "  ",  it)


for es in range(iter):

# construct one solution
  th1=np.zeros(cl_num) 

  for i in range(cl_num):       # generate random threshold 
    th1[i]=random.uniform(0, 1)

  mask=np.where(th1<=th)     # compare with initial threshold and select target clusters
  mask=mask[1]+1 
  
   
# construct one solution
  tempset=np.zeros(len(mask))
  for i in range(len(mask)):
    # tempset[i]=np.array(random.choices(cln.labels[cln.labels.eq(mask[i])].index))
    tempset[i]=cln.labels[cln.labels.eq(mask[i])].sample().index.values
#construct the respective matrix
  matn=np.zeros((df.shape[0],len(mask)+1))
# df_ = pd.DataFrame(index=61, columns=cl_num+1)
  for i in range(len(mask)):
     matn[:,i]=Xn.iloc[:,int(tempset[i])]
  matn[:,len(mask)]=yn
  
  Xnew=matn[:,0:len(mask)]
  ynew=matn[:,len(mask)]
  
  p_final=np.zeros(10)
  r_final=np.zeros(10)
  f_final=np.zeros(10)
  s=np.zeros(10)
  nm=np.zeros(10)
  for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(Xnew,ynew, test_size=0.2)
        svm_linear=svm.SVC(C=50,kernel="linear")
        svm_linear.fit(X_train,y_train)
        s[i]=svm_linear.score(X_test,y_test)
        # dectree = tree.DecisionTreeClassifier()
        # dectree.fit(X_train,y_train)
        # s[i]=dectree.score(X_test,y_test)
        # nm[i]=normalized_mutual_info_score(dectree.predict(X_test),y_test)

        cm=confusion_matrix(y_test,svm_linear.predict(X_test))
        p_final[i]= cm[1,1] / (cm[0,1]+cm[1,1])
        r_final[i]= cm[1,1] / (cm[1,1]+cm[1,0])
  pre=np.mean(p_final) 
  rec=np.mean(r_final) 
  fscore=fscore=2 * ((pre*rec) / (pre+rec))
  tempacc=np.mean(s)
  # tempnm=np.mean(nm)
  it=it+1
  if tempacc > acc:
     opt=opt+1
     acc=tempacc
#     nmi=tempnm
     pr=pre
     re=rec 
     fs=fscore 
     accC[es] = acc 
     count=count+1   
     s_size[count]=len(mask)
     XFinal=Xnew
     YFinal=ynew

     m=mask-1    
     for i in range(len(m)):       # generate random threshold 
        th[0,m[i]]=th[0,m[i]]+(alfa*(1-th[0,m[i]]))
        if (th[0,m[i]]>1):
            th[0,m[i]]=1
            
     print ("acc",  "  ", acc,"      ","subset_size", "  ", s_size[count],"     ", "iteration", "  ",  it)
     
  elif tempacc == acc and len(mask)<s_size[count]:
      
     count=count+1   
     s_size[count]=len(mask) 
     
     #    
     print ("acc",  "  ", acc,"      ","subset_size", "  ", s_size[count],"     ", "iteration", "  ",  it)
      
  else:
     accC[es] = acc
     print ("acc",  "  ", acc,"      ","subset_size", "  ", s_size[count],"     ", "iteration", "  ",  it)
t1= time()   

t1-t0  
####################################################################################


