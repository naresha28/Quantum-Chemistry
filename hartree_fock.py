#!/usr/bin/env python
# coding: utf-8

# In[1]:

"""
This code calculates the energies of chemical systems using the following input information:

- Molecular orbital coefficients: s.dat
- Two-electron integrals: t.dat
- Potential Integrals: v.dat
- Nuclear repulsion energy: enuc.dat
- Electron repulsion integrals: eri.dat

"""


# Import required libraries
import numpy as np
import math

# Read the input files containing electronic structure data
# Each file contains data related to specific aspects of the electronic structure

s_org = np.loadtxt('s.dat')
t_org = np.loadtxt('t.dat')
v_org = np.loadtxt('v.dat')
enuc = np.loadtxt('enuc.dat')
eri_org = np.loadtxt('eri.dat')

# Read the number of basis sets (nb) and occupancies (occ) from 'data.dat'
# 'nb' represents the total number of basis sets used in the calculation
# 'occ' contains information about the occupancies of the basis sets
nb, occ = np.loadtxt('data.dat', dtype=int)



# In[2]:


print(nb,occ)


# In[3]:


# Initialize matrices to store electronic structure data and intermediate results for the electronic structure calculation.
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


# Arrange the data to match 0-based indexing
s_org[:,[0,1]] = s_org[:,[0,1]]-1
t_org[:,[0,1]] = t_org[:,[0,1]]-1
v_org[:,[0,1]] = v_org[:,[0,1]]-1
eri_org[:,[0,1,2,3]] = eri_org[:,[0,1,2,3]]-1


# In[5]:


# Store the data from input files into matrices representing relevant electronic structure integrals.

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
#print("S matrix")
#print(S_mat)
#print("V matrix")
#print(V_mat)
#print("T matrix")
#print(T_mat)
#print("H matrix")
#print(H_mat)


# In[7]:



"""
Compute the spin-half transformation matrix.

"""
    # Compute the eigenvalues (w) and eigenvectors (v) of the input matrix (S_mat).
w, v = np.linalg.eigh(S_mat)
daig_half = np.diag(w ** (-0.5))
S_half = np.dot(v, np.dot(daig_half, np.transpose(v)))




# In[8]:


def Fock_matrix(H_mat,nb,P):
    """
    Calculate the Fock matrix based on the given input matrices and number of basis sets.

    Parameters:
        H_mat : The core Hamiltonian matrix.
        nb : The number of basis sets.
        P : The density matrix.
        Eri_mat: The electron repulsion integral matrix.

    Returns:
         The Fock matrix.
    """

    for i in range(nb):
        for j in range(nb):
            F[i,j]=H_mat[i,j]
            for k in range(nb):
                for l in range (nb):
                    F[i,j] = F[i,j]+P[k,l]*(2*Eri_mat[i,j,k,l]-Eri_mat[i,k,j,l])
    return F


# In[9]:


def Density(C,den_mat,nb,occ,old_den):
    """
    Calculate the density matrix based on molecular orbital coefficients (C), number of basis sets (nb),
    number of occupied orbitals (occ), and the previous iteration's density matrix (old_den).


    Returns:
     The updated density matrix (den_mat) and the previous density matrix (old_den).
    """
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
    """
    Calculate the total energy of a chemical system based on the density matrix, Fock matrix, and core Hamiltonian.

    Returns:
         The total energy of the chemical system.
    """
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

"""
SCF loop starts here.

The loop iterates a maximum of 100 times to obtain a self-consistent solution for the electronic structure of the chemical system.
It calculates the Fock matrix, updates the density matrix, computes the total energy,
and checks for convergence in each iteration.
If convergence is reached (both del_e and del_D are below the specified thresholds), the loop is terminated, and the self-consistent solution is obtained.

"""
#print('Iteration', 'total energy','energy diff', 'density difference' )
print(f"{'Iteration':<10}{'total_energy':<20}{'energy_diff':<20}{'density_diff':<20}")

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
    #print("New energy", new_e)
#    print(new_e)
    tot_e = new_e + enuc
    del_D = rms_D(den_mat,old_den,nb)
    del_e = abs(new_e-old_e)
    old_e = new_e
    #print(i, tot_e, del_e, del_D)
    print(f"{i:<10}{tot_e:<20.12f}{del_e:<20.12f}{del_D:<20.12f}")
    if(del_e < convergence and del_D < eps_d):
        print("Loop is exited because convergence is achieved")
        break


# In[13]:


#print("Eigen value matrix")
#print(w)
#print("Diagonal matrix")
#print("Diagonal_half matrix")
#print(daig_half)
#print(S_half)
#print("Fock Matrix")
#print(F)
#print("Eigenvectors of FP Matrix")
#print(C)
#print("denisty matrix")
#print(den_mat)
#print(tot_e)
#print(enuc)

print('Energy of the chemical system is:',tot_e, 'eV')

# In[ ]:
