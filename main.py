import numpy as py
import math
from numpy import linalg as la
L = 100 #longueur de la barre
N=100
h=L/N #subdivision
T0=30.0 # température à x=0
Tl=50.0 #température à x=L


#pour cette méthode on cherche la solution de l'équation de la chaleur en 2D
#on a réussi à écrire ce système sous la forme h(x)=1/2<A,X>-<b,X>+c=0
#dont on doit résoudre AX-B=0 avec la méthode de newtone
#d'où on cherche le zéro de grad(h(X))
#initialisation de A
def A (N):
    B = py.zeros(((N-1), (N-1)), dtype=int)        #initialiser la matrice B, ces matrices constituent la diagonale de A
    for i in range(N-1):
        for j in range(N-1):
            if (j == (i-1)) or (j == (i+1)):
                B[i, j] = -1                    #remplir les diagonales superieure et inferieure avec des -1
            elif i == j:
                B[i, j] = 4                     #remplir la diagonale avec des 4

    I = py.identity(N-1, dtype=int)                           #definir la matrice identité de dimensions N-1
    Z = py.zeros((N-1, N-1), dtype=int)                       #definir la matrice nulle de dimensions N-1
    A = py.zeros(((N-1)**2, (N-1)**2), dtype=int)  #inittialiser A avec des 0
    for i in range(N-1):
        if (i==0):                              #créer la premiere partie de A, une partie de A est une matrice (N-1)*((N-1)**2)
            L = py.concatenate((B, -I), axis=1)            #concatener B,-I et plusieurs Z jusqu'à completer la partie
            for k in range(N-3):
                L = py.concatenate((L, Z), axis=1)
        elif (i==N-2):                          #créer la derniere partie de A
            L = Z
            for k in range(N - 4):
                L = py.concatenate((L, Z), axis=1)
            L = py.concatenate((L, -I, B), axis=1)
        elif (i==1):                            #créer la deuxieme partie de A
            L =py.concatenate((-I, B, -I), axis=1)
            for k in range(N-4):
                L = py.concatenate((L, Z), axis=1)
        else:                                   #créer une partie du milieu de A
            L = Z
            for k in range(i-2):
                L = py.concatenate((L, Z), axis=1)
            L = py.concatenate((L, -I, B, -I), axis=1)
            for k in range(i+2, N-1):
                L = py.concatenate((L, Z), axis=1)
        A = py.concatenate((A, L))         #concatener partie par partie jusqu'a former A
    A = A[(N-1)**2:,:]                  #enlever les 0 ajoutés à l'initialisation
    return A/h**2

#initialisation de b
def b(g,T0j, TNj, Ti0, TiN, N, h, x, y):
    b = py.zeros(((N-1)**2))
    for i in range(N-1):
        C = py.zeros((N-1))                #C est une partie de b, C est de dimensions (n-1)*1
        if (i==0):                      #créer la premiere partie
            C[0]=g(x[0],y[0])+ T0j[0]/h**2 + Ti0[0]  #C[0]= g(x1,y1) + T01/h**2 + T10/h**2
            C[N-2]=g(x[N-2],y[0])+ TNj[0]/h**2
            for k in range(1, N-2):
                C[k]=g[x[k],y[0]]+ Ti0[k]/h**2       #C[i]= g(xi,y1) + Ti0/h**2
        elif (i==(N-2)):                #créer la derniere partie
            C[0]=g(x[0],y[N-2])+ TiN[0]/h**2
            C[N-2]=g(x[N-2],y[N-2])+ TNj[N-2]/h**2 + TiN[N-2]/h**2  #C[N-2]= g(xN-1,yN-1) + TN,N-1/h**2 + TN-1,N/h**2
            for k in range(1, N-2):
                C[k]=g[x[k],y[N-2]]+ TiN[k]/h**2
        else:                           #créer une partie du milieu de b
            C[0]=g(x[0],y[i])+ T0j[i]/h**2
            C[N-2]=g(x[N-2],y[i])+ TNj[i]/h**2
            for k in range(1, N-2):
                C[k]=g[x[k],y[i]]
        b = py.concatenate((b, C))         #concatener partie par partie jusqu'a former b
    b = b[(N-1)**2:,:]                  #enlever les 0 ajoutés à l'initialisation
    return b
def g(m,n): # on prend par exemple g=sin
    g=math.sin(m+n)
    return g

#######################################

def newton (a,epsilone): #epsilone ets l'erreur d'aproximation
    delta=1
    n=0 #nombre d'itération
    # on choisit delta= 1 qui est strictement supérieure à epsilone pour pouvoir exécuter la 1ere itération de l'algorithme
    while delta>epsilone: #condition d'arret
        x = -(py.matmul(f(a,A,b),j(a)))+a
        delta=la.norm(x-a)
        a=x
        n+=1
        if (n>100):
            print("l'algorithme ne converge pas")
            break
    return x,delta,n


def f(x,A,b):         # f= Ax-b
    return py.matmul(A,x)-b

def j(x): # déterminer l'inverse de la matrice jacobienne
    return py.inv(py.jacobian(py.matmul((A, x)-b)))

y= py.random.rand((N-1)**2) #y est un vecteur aléatoire

print(newton(y,0.1))