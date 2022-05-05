#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Read the input files
import numpy as np
import math
#### Read the inputfiles    
s_org = np.loadtxt('s.dat')
t_org = np.loadtxt('t.dat')
v_org = np.loadtxt('v.dat')
enuc = np.loadtxt('enuc.dat')
eri_org = np.loadtxt('eri.dat')
### Read number of basis sets and occupancies
nb,occ = np.loadtxt('data.dat',dtype=int)


# In[2]:


print(nb,occ)


# In[3]:


# Define matrices
S_mat = np.zeros((nb,nb))
V_mat = np.zeros((nb,nb))
T_mat = np.zeros((nb,nb))
H_mat = np.zeros((nb,nb))
Eri_mat = np.zeros((nb,nb,nb,nb))
F = np.zeros((nb,nb))
den_mat = np.zeros((nb,nb))
old_den = np.zeros((nb,nb))
old_e = 0.0


# In[4]:


## Arrange the data
s_org[:,[0,1]] = s_org[:,[0,1]]-1
t_org[:,[0,1]] = t_org[:,[0,1]]-1
v_org[:,[0,1]] = v_org[:,[0,1]]-1
eri_org[:,[0,1,2,3]] = eri_org[:,[0,1,2,3]]-1


# In[5]:


## store the data from input files into matrices
for row in s_org:
           
    S_mat[int(row[0]),int(row[1])] = row[2]
    S_mat[int(row[1]),int(row[0])] = row[2]
    
for row in t_org:
    T_mat[int(row[0]),int(row[1])] = row[2]
    T_mat[int(row[1]),int(row[0])] = row[2]
for row in v_org:
    V_mat[int(row[0]),int(row[1])] = row[2]
    V_mat[int(row[1]),int(row[0])] = row[2]
for row in eri_org:
    Eri_mat[int(row[0]),int(row[1]),int(row[2]),int(row[3])] = row[4]
    Eri_mat[int(row[1]),int(row[0]),int(row[2]),int(row[3])] = row[4]
    Eri_mat[int(row[0]),int(row[1]),int(row[3]),int(row[2])] = row[4]
    Eri_mat[int(row[1]),int(row[0]),int(row[3]),int(row[2])] = row[4]
    Eri_mat[int(row[2]),int(row[3]),int(row[0]),int(row[1])] = row[4]
    Eri_mat[int(row[3]),int(row[2]),int(row[0]),int(row[1])] = row[4]
    Eri_mat[int(row[2]),int(row[3]),int(row[1]),int(row[0])] = row[4]
    Eri_mat[int(row[3]),int(row[2]),int(row[1]),int(row[0])] = row[4]


# In[6]:


print(S_mat.shape)
H_mat = T_mat + V_mat
#### print  matrices
print("S matrix")
print(S_mat)
print("V matrix")
print(V_mat)
print("T matrix")
print(T_mat)
print("H matrix")
print(H_mat)


# In[7]:


#### Spin_half matrix
w,v = np.linalg.eigh(S_mat)
daig_half = np.diag(w**(-0.5))
S_half = np.dot(v,np.dot(daig_half,np.transpose(v)))


# In[8]:


def Fock_matrix(H_mat,nb,P):
    
    for i in range(nb):
        for j in range(nb):
            F[i,j]=H_mat[i,j]
            for k in range(nb):
                for l in range (nb):
                    F[i,j] = F[i,j]+P[k,l]*(2*Eri_mat[i,j,k,l]-Eri_mat[i,k,j,l])                
    return F


# In[9]:


def Density(C,den_mat,nb,occ,old_den):
    old_den = np.zeros((nb,nb))
    for i in range (nb):
        for j in range(nb):
            old_den[i,j] = den_mat[i,j]
            den_mat[i,j] = 0.0           
            for m in range (occ):
                den_mat[i,j] = den_mat[i,j]+C[i,m]*C[j,m]
                
                
    return den_mat, old_den


# In[10]:


def energy(den_mat, F, H_mat, nb):
    EN = 0.0
    for i in range (nb):
        for j in range(nb):
            EN = EN+(den_mat[i,j]*(H_mat[i,j]+F[i,j]))
            
    return EN


# In[11]:


#### difference between the denisities 
def rms_D(D2,D1,nb):
    sums = 0.0
    for i in range(nb):
        for j in range (nb):
            sums = sums+((D2[i,j]-D1[i,j])**2)
    rms_D = math.sqrt(sums)
    return rms_D


# In[12]:


convergence = 1e-9
eps_d = 1e-8
###### SCF loop starts here
for i in range(100):    
    F = Fock_matrix(H_mat,nb,den_mat)
    XT = np.transpose(S_half)
    FX = np.dot(F,S_half)
    FP = np.dot(XT,FX)
    eig,CP = np.linalg.eigh(FP)
    C= np.dot(S_half,CP)
    
    den_mat,old_den = Density(C,den_mat,nb,occ,old_den)
    
#old_den = den_mat
    new_e = energy(den_mat, F, H_mat, nb)
    print("New energy")
    print(new_e)
    tot_e = new_e + enuc
    del_D = rms_D(den_mat,old_den,nb)
    del_e = abs(new_e-old_e)
    old_e = new_e
    print(i, "total energy", tot_e, del_e, del_D)
    if(del_e < convergence and del_D < eps_d):
        print("convergence is reached")
        break


# In[13]:


print("Eigen value matrix")
print(w)
#print("Diagonal matrix")
#print(daig)
print("Diagonal_half matrix")
print(daig_half)
print(S_half)
print("Fock Matrix")
print(F)
print("Eigenvectors of FP Matrix")
print(C)
print("denisty matrix")
print(den_mat)
print(tot_e)
print(enuc)


# In[ ]:




