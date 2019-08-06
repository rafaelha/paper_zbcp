import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
mpl.use('Agg')
import numpy as np
import time
from numpy import sin, cos 
import scipy.sparse as sp
from scipy.sparse import kron
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
import os
import glob
import pickle

def kr(a, b, c): # Kronecker product of three matrices
    return kron(a, kron(b, c))

s0 = sp.csc_matrix([[1, 0], [0, 1]]); t0 = s0; sig0 = s0;
sx = sp.csc_matrix([[0, 1], [1, 0]]); tx = sx; sigx = sx;
sy = sp.csc_matrix([[0, -1j], [1j, 0]]); ty = sy; sigy = sy;
sz = sp.csc_matrix([[1, 0], [0, -1]]); tz = sz; sigz = sz;

z00 = sp.csc_matrix(kr(sigz, t0, s0).toarray())
zzz = sp.csc_matrix(kr(sigz, tz, sz).toarray())
zzx = sp.csc_matrix(kr(sigz, tz, sx).toarray())
zzy = sp.csc_matrix(kr(sigz, tz, sy).toarray())
x00 = sp.csc_matrix(kr(sigx, t0, s0).toarray())

ax = 20 #lattice constant Angstrom
ay = ax; az = ax;
ax2 = ax**2 # material parameters: Cano et al. (Bernevig group) PRB 95 161306 (2017)
ay2 = ay**2; az2 = az**2;
C0 = -0.0145
C1 = 10.59
C2 = 11.5
A = 0.889
M0 = 0.0205
M1 = -18.77
M2 = -13.5
mu = C0 - C1 * M0 / M1 # tune chemical potential into Dirac nodes

Bx = 0.035
Bz = 0.00
gap = 0.035
g = 1
mu_local =-28e-3

Lx = 20 
Ly = 100 
Lmag = 10

delta = 0.01 

jx = np.zeros((4,4))
#jx[0,0] = Bx**2 / gap
jx[0,3] = - Bx**3 / gap**2
jx[1,2] = -0.5 * g * Bx
jx[2,1] = -0.5 * g * Bx
jx[3,0] = -Bx**3 / gap**2
#jx[3,3] = Bx **2 / gap
jz = np.diag(np.array([3/2,1/2,-1/2,-3/2], dtype='complex128'))

def n2(x, y): # map 2d real space to 1d vector-indices
    return int(Lx * y + x) 
def build_hop():
    size = (Lx * Ly, Lx * Ly)
    hx_obc = sp.lil_matrix(size)
    hy_obc = sp.lil_matrix(size)
    hx_pbc = sp.lil_matrix(size)
    hy_pbc = sp.lil_matrix(size)
    dist = np.zeros(Lx * Ly)
    for i in np.arange(Lx):
        for j in np.arange(Ly):
            n0 = n2(i, j)
            if i < Lx - 1:
                hx_obc[n0, n2(i+1,j)] = 1
            if j < Ly - 1:
                hy_obc[n0, n2(i,j+1)] = 1
            hx_pbc[n0, n2((i+1)%Lx,j)] = 1
            hy_pbc[n0, n2(i,(j+1)%Ly)] = 1

            #dist[n2(i, j)] = np.sqrt((i- (Lx-1)/2)**2 + (j - (Ly-1)/2)**2)
            #dist[n2(i, j)] = np.sqrt((i)**2 + (j)**2)
            dist[n2(i, j)] = np.abs(i -(Lx-1)//2)
    return  hx_obc.tocsc(), \
            hy_obc.tocsc(), \
            hx_pbc.tocsc(), \
            hy_pbc.tocsc(), \
            sp.diags(np.ones(Lx * Ly)).tocsc(), \
            np.kron(dist, np.ones(8))
def buildDelta(delta):
    DD = sp.lil_matrix((Lx * Ly, Lx * Ly))
    TT = sp.lil_matrix((Lx * Ly, Lx * Ly), dtype='complex128')
    Tmu = sp.lil_matrix((Lx * Ly, Lx * Ly), dtype='complex128')
    for i in [0,Lx-1]:
        for j in np.arange(3*Ly//4,Ly):
            DD[n2(i, j), n2(i, j)] = delta
            #Tmu[n2(i, j), n2(i, j)] = 1
            #TT[n2(i, j), n2(i, j)] = 1
        for j in np.arange(Ly//4):
            DD[n2(i, j), n2(i, j)] = delta
            #Tmu[n2(i, j), n2(i, j)] = 1
            #TT[n2(i, j), n2(i, j)] = 1
        for j in np.arange(Ly//4, 3*Ly//4):
            if i == 0:
                DD[n2(i, j), n2(i, j)] = delta
                #Tmu[n2(i, j), n2(i, j)] = 1
            else:
                #DD[n2(i, j), n2(i, j)] = -delta
                #DD[n2(i, j), n2(i, j)] = -delta
                TT[n2(i, j), n2(i, j)] = 1
                #DD[n2(i, j), n2(i, j)] = delta
                Tmu[n2(i, j), n2(i, j)] = 1
        """
        for j in np.arange(Lmag):
            TT[n2(i, j), n2(i, j)] = 1
        for j in np.arange(Lmag, Ly):
            DD[n2(i, j), n2(i, j)] = delta
        """
    return DD.tocsc(), TT.tocsc(), Tmu.tocsc()

def buildHk(kz, D, hx, hy, di2, TT, Tmu):
    #hoppings
    H =  kron(hx, (-C2/ax2) * z00 - M2/ax2 * zzz - A * 1j/(2 * ax) * zzx)
    H += kron(hy, (-C2/ay2) * z00 - M2/ay2 * zzz - A * 1j/(2 * ay) * zzy)
    H += H.T.conj()
    # on-site potential
    H += kron(di2, (C0 + 2*C1/az2*(1 - cos(kz*az)) + C2*(2/ax2 + 2/ay2) - mu) * z00\
              +(M0 + 2*M1/az2*(1 - cos(kz*az)) + M2*(2/ax2 + 2/ay2)) * zzz) \
            + kron(D, x00)

    H += kron(TT, kron(sig0, jx)) + kron(TT, Bz * kron(sig0, jz)) # magnetic field
    H += kron(TT, kron(sigz, kron(s0,s0))) * mu_local
    return H
def plotDOS(spectrum, bins=300, range=(-20,20)):
    plt.ion()
    plt.figure(figsize=(5,4))
    plt.hist(spectrum.reshape((spectrum.size,)) * 1000, range=range, bins=bins)
    plt.xlabel('Energy $E$')
    plt.ylabel('dos')
    plt.xlim(range)
    plt.tight_layout()
def diagHk(kz, delta=0.1, sparse=True, k=100):

    hx2_obc, hy2_obc, hx2_pbc, hy2_pbc, di2, dist2 = build_hop()
    hx = hx2_obc
    hy = hy2_pbc
    #hy = hy2_obc #use open boundary conditions. Dima's paper used PBC in y  

    D2, TT, Tmu = buildDelta(delta)
    if sparse:
        nn = k    
    else:
        nn = 8 * Lx * Ly

    spectrum = np.zeros((nn, len(kz)))
    pos = np.zeros((nn, len(kz)))
    vectors = np.zeros((8 * Lx * Ly, nn, len(kz)), dtype='complex128')

    vecs_old = np.zeros((8 * Lx * Ly, nn))
    second = False

    for j in np.arange(len(kz)):
        H = buildHk(kz[j], D2, hx, hy, di2, TT, Tmu)
        if sparse:
            ev, vecs = sp.linalg.eigsh(sp.csc_matrix(H), k=k, sigma=0)
            ev = np.real(ev)
            idx = ev.argsort()[::-1]
            ev = ev[idx]
            vecs = vecs[:,idx]
        else:
            ev, vecs = np.linalg.eigh(H.toarray())
        if second:
            idx = np.argmax(np.abs(vecs_old.T.conj().dot(vecs)), axis=1)
            ev = ev[idx]
            vecs = vecs[:,idx]
        
        #second = True
        vecs_old = np.array(vecs)
            
        p = (np.abs(vecs.T)**2).dot(dist2)
        pos[:, j] = p
        spectrum[:, j] = ev        
        vectors[:,:,j] = vecs
        print(j)

    return spectrum, pos, vectors
def plotHk(kz, spectrum, pos, delta=0.1, sparse=True, k=100, ylim=(-20,20)):

    nn = spectrum.shape[0]
    plt.ion()    

    fig, ax = plt.subplots(1,1, num="one")
    fig.set_size_inches(5,4)

    norm = plt.Normalize(np.min(pos), np.max(pos))

    for i in np.arange(nn):
        y = spectrum[i,:] * 1000
        y = np.append(y, np.flip(y, 0))
        
        points = np.array([np.append(kz*az, np.flip(-kz*az, 0)), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cdata = np.append(pos[i,:-1], np.flip(pos[i,1:], 0))

        lc = LineCollection(segments, cmap='cool', norm=norm)
        # Set the values used for colormapping
        lc.set_array(cdata)
        lc.set_linewidth(0.8)
        line = ax.add_collection(lc)


    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Expectation distance from center')
    ax.set_ylim(ylim)
    plt.xlabel('$k_z a$')
    plt.ylabel('$E$ [meV]')
    plt.xlim((-np.pi/2, np.pi/2))
    #plt.savefig('new_code_bandstructure.pdf')

def next(name):
    i = 0
    while os.path.exists(name % i):
        i += 1 
    return name % i
def nextrun(name):
    i = 50
    while os.path.exists(name % i):
        i += 10 
    return i, name % i
def savefig():
    path = "C:/Users/Rafael/Desktop/Dirac_NS_junction/_notes/Bfig/res%s.pdf"
    plt.savefig(next(path))
if __name__ == '__main__':
    plt.close('all')

    start_time = time.time()

    kz = np.linspace(-np.pi/2/2/az, 0/az, 600) 
    k = 140 #number of energies to calculate from sparse diag
    spectrum, pos, vecs = diagHk(kz, delta=delta, sparse=True, k=k)    
    plotHk(kz, spectrum, pos, ylim=(-20,20))
    plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)+', $\Delta$='+str(round(delta*1e3,2))\
              +'meV, $B_x$='+str(Bx)+', gp='+str(gap)+', t='\
              +str(np.round(time.time()-start_time))+'s')
    plt.xlim((-np.max(np.abs(kz))*az, np.max(np.abs(kz))*az))
    plt.tight_layout()
    plt.ylim((-10,10))
    #savefig()

    f1 = open('plotdata2.pickle', 'ab')
    pickle.dump(Lx, f1)
    pickle.dump(Ly, f1)
    pickle.dump(delta, f1)
    pickle.dump(Bx, f1)
    pickle.dump(gap, f1)
    pickle.dump(kz, f1)
    pickle.dump(spectrum, f1)
    pickle.dump(pos, f1)
    #pickle.dump(vecs, f1)
    pickle.dump(0, f1)

    f1.close()
