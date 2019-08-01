import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import kwant as kw
import numpy as np
import tinyarray
from numpy import kron
import time
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from numpy.random import rand
import pickle
import datetime as dt
import sys
import os

s0 = tinyarray.array([[1, 0], [0, 1]]); t0 = s0; sig0 = s0;
sx = tinyarray.array([[0, 1], [1, 0]]); tx = sx; sigx = sx;
sy = tinyarray.array([[0, -1j], [1j, 0]]); ty = sy; sigy = sy;
sz = tinyarray.array([[1, 0], [0, -1]]); tz = sz; sigz = sz;

def kr(a, b, c): # Kronecker product of three matrices
    return kron(a, kron(b, c))

ax = 20 #lattice constant Angstrom
ax2 = ax**2 # material params Cano et al. (Bernevig) PRB 95 161306 (2017)
ay = 20; az = 20; ay2 = ay**2; az2 = az**2;
C0 = -0.0145
C1 = 10.59
C2 = 11.5
A = 0.889
M0 = 0.0205
M1 = -18.77
M2 = -13.5
B1 = 0

# Magnetic field
Bx = 0.035 
ggap = 0.035
g = 1
mu_local =-28e-3 # local chemical potential to push magnetic ggap to fermi level
jx = np.zeros((4,4))
#jx[0,0] = Bx**2 / gap
jx[0,3] = - Bx**3 / ggap**2
jx[1,2] = -0.5 * g * Bx
jx[2,1] = -0.5 * g * Bx
jx[3,0] = -Bx**3 / ggap**2
#jx[3,3] = Bx **2 / ggap
#######################################
############### parameters ############
#######################################
delta = 0.0001 #0.00018*0.5 # SC order parameter (real)
phi = np.pi
mu = (C0 - C1 * M0 / M1) #- 7e-3# tune chem pot to Dirac nodes
W = 0.005 # disorder strength
Lx = 15
Ly = 300
Lz = 15

xdir = 0 #set direction of transport to one
ydir = 1
zdir = 0

def onsite(): # onsite energy without gap
    return kr(sigz, t0, s0) * (C0 + 2*C1/az2 + 2*C2/ax2 + 2*C2/ay2 - mu)\
            + kr(sigz, tz, sz) * (M0 + 2*M1/az2 + 2*M2/ax2 + 2*M2/ay2)
def gap(): # onsite energy with gap
    return kr(sigx, t0, s0) * delta
def gap_t(x, y, z): # onsite energy with gap
    d = 0
    if x == 0:
        d = delta
    return kr(sigx, t0, s0) * d
def gap_b(x, y, z): # onsite energy with gap
    d = 0
    if x == Lx-1:
        d = delta
    return kr(sigx, t0, s0) * d
def disorder(): # onsite energy with disorder in uniform interval [-W/2, W/2]
    return (rand() - 1/2) * W * kr(sigz, t0, s0)
def disorder_t(x, y, z): # onsite energy with disorder in uniform interval [-W/2, W/2]
    d = 0
    if x == 0:
        d = W 
    return (rand() - 1/2) * d * kr(sigz, t0, s0)
def disorder_b(x, y, z): # onsite energy with disorder in uniform interval [-W/2, W/2]
    d = 0
    if x == Lx-1:
        d = W 
    return (rand() - 1/2) * d * kr(sigz, t0, s0)
def mag(x, y, z): # Zeeman field
    return kron(sig0, jx) + kr(sigz, t0, s0) * mu_local
def mag_t(x, y, z): # Zeeman field
    d = 0
    if x == 0:
        d = 1 
    return ( kron(sig0, jx) + kr(sigz, t0, s0) * mu_local ) * d
def mag_b(x, y, z): # Zeeman field
    d = 0
    if x == Lx-1:
        d = 1 
    return ( kron(sig0, jx) + kr(sigz, t0, s0) * mu_local ) * d

def DOS(sys, k=100, range=(-1.5*50e-6,1.5*50e-6), bins=1000, fignum=2): # plot the lowest eigenvalues in a hist-plot
    H = sp.csc_matrix(sys.hamiltonian_submatrix(sparse=True))
    ev, _ = eigsh(H, k=k, sigma=0)

    plt.ion()
    #plt.figure()
    #plt.plot(ev,'.')
    plt.figure()
    plt.clf()
    plt.hist(ev, range=range, bins=bins)
    plt.xlim(range)
    plt.xlabel('Energy')
    plt.ylabel('Number of states')
    return ev
def build_sys():
    lat = kw.lattice.general([(ax,0,0), (0,ay,0), (0,0,az)], norbs=8)
    sys = kw.Builder()
    sym_left = kw.TranslationalSymmetry((-1*ax*xdir, -1*ay*ydir, -1*az*zdir))
    sym_right = kw.TranslationalSymmetry((1*ax*xdir, 1*ay*ydir, 1*az*zdir))
    lead0 = kw.Builder(sym_left, conservation_law=-kr(sigz,t0,s0))
    lead1 = kw.Builder(sym_right)

    ################# onsite #############################################
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for i in np.arange(Lx * (1-xdir) + xdir):
        for j in np.arange(Ly * (1-ydir) + ydir):
            for k in np.arange(Lz * (1-zdir) + zdir):
                lead0[lat(i, j, k)] = onsite() #+ mag_b(i, j, k)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # on site potential
    for i in np.arange(Lx):
        for j in np.arange(Ly):
            for k in np.arange(Lz):
                sys[(lat(i, j, k) )] = onsite() + disorder()
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for i in np.arange(Lx * (1-xdir) + xdir):
        for j in np.arange(Ly * (1-ydir) + ydir):
            for k in np.arange(Lz * (1-zdir) + zdir):
                lead1[lat(i, j, k)] = onsite() + gap()



    ################# hopping ############################################
    ######################################################################
    #hBx = 1j * B1 / (ax2 * az) * kr(sigz, tx, s0)
    #hBy = -1j * B1 / (ay2 * az) * kr(sigz, tx, s0)

    hx = kr(sigz, t0, s0) * (-C2)/ax2\
            + kr(sigz, tz, sz) * (-M2)/ax2\
            + kr(sigz, tz, sx) * A * (-1j)/(2 * ax)
    hy = kr(sigz, t0, s0) * (-C2)/ay2\
            + kr(sigz, tz, sz) * (-M2)/ay2\
            + kr(sigz, tz, sy) * A * (-1j)/(2 * ay)
    hz = kr(sigz, t0, s0) * (-C1)/az2\
            + kr(sigz, tz, sz) * (-M1)/az2
    #        - (hBx + hBy)

    sys[kw.builder.HoppingKind((1, 0, 0), lat, lat)] = hx
    sys[kw.builder.HoppingKind((0, 1, 0), lat, lat)] = hy
    sys[kw.builder.HoppingKind((0, 0, 1), lat, lat)] = hz

    """
    sys[kw.builder.HoppingKind((1, 0, 1), lat, lat)] = hBx
    sys[kw.builder.HoppingKind((-1, 0, 1), lat, lat)] = hBx
    sys[kw.builder.HoppingKind((0, 1, 1), lat, lat)] = hBy
    sys[kw.builder.HoppingKind((0, -1, 1), lat, lat)] = hBy
    """
    # PBC along z
    for i in np.arange(Lx):
        for j in np.arange(Ly):
            sys[lat(i, j, 0), lat(i, j, Lz-1)] = hz
    """
            if i < Lx-1:
                sys[lat(i+1, j, 0), lat(i, j, Lz-1)] = hBx
            if i > 0:
                sys[lat(i-1, j, 0), lat(i, j, Lz-1)] = hBx
            if j < Ly-1:
                sys[lat(i, j+1, 0), lat(i, j, Lz-1)] = hBy
            if j > 0:
                sys[lat(i, j-1, 0), lat(i, j, Lz-1)] = hBy
    """

    ############### lead0 ##############
    lead0[kw.builder.HoppingKind((1, 0, 0), lat, lat)] = hx
    lead0[kw.builder.HoppingKind((0, 1, 0), lat, lat)] = hy
    lead0[kw.builder.HoppingKind((0, 0, 1), lat, lat)] = hz
    """
    lead0[kw.builder.HoppingKind((1, 0, 1), lat, lat)] = hBx
    lead0[kw.builder.HoppingKind((-1, 0, 1), lat, lat)] = hBx
    lead0[kw.builder.HoppingKind((0, 1, 1), lat, lat)] = hBy
    lead0[kw.builder.HoppingKind((0, -1, 1), lat, lat)] = hBy
    """
    # PBC along z
    for i in np.arange(Lx):
        lead0[lat(i, 0, 0), lat(i, 0, Lz-1)] = hz
    """
        if i < Lx-1:
            lead0[lat(i+1, 0, 0), lat(i, 0, Lz-1)] = hBx
        if i > 0:
            lead0[lat(i-1, 0, 0), lat(i, 0, Lz-1)] = hBx
        lead0[lat(i, 1, 0), lat(i, 0, Lz-1)] = hBy
        lead0[lat(i, -1, 0), lat(i, 0, Lz-1)] = hBy
    """
    ############### lead1 ##############
    lead1[kw.builder.HoppingKind((1, 0, 0), lat, lat)] = hx
    lead1[kw.builder.HoppingKind((0, 1, 0), lat, lat)] = hy
    lead1[kw.builder.HoppingKind((0, 0, 1), lat, lat)] = hz
    """
    lead1[kw.builder.HoppingKind((1, 0, 1), lat, lat)] = hBx
    lead1[kw.builder.HoppingKind((-1, 0, 1), lat, lat)] = hBx
    lead1[kw.builder.HoppingKind((0, 1, 1), lat, lat)] = hBy
    lead1[kw.builder.HoppingKind((0, -1, 1), lat, lat)] = hBy
    """
    # PBC along z
    for i in np.arange(Lx):
        lead1[lat(i, 0, 0), lat(i, 0, Lz-1)] = hz
    """
        if i < Lx-1:
            lead1[lat(i+1, 0, 0), lat(i, 0, Lz-1)] = hBx
        if i > 0:
            lead1[lat(i-1, 0, 0), lat(i, 0, Lz-1)] = hBx
        lead1[lat(i, 1, 0), lat(i, 0, Lz-1)] = hBy
        lead1[lat(i, -1, 0), lat(i, 0, Lz-1)] = hBy
    """

    sys.attach_lead(lead0)
    sys.attach_lead(lead1)

    #kw.plot(sys)
    sys = sys.finalized()
    return sys, lead0, lead1
def sim(sys, range, plot=False):
    n = len(range)
    energies = range#np.linspace(range[0], range[1], n)
    N = np.zeros(n)
    Ree = np.zeros(n)
    Reh = np.zeros(n)
    #G2 = np.zeros(n)

    for i in np.arange(n):
        smatrix = kw.smatrix(sys, energies[i])

        N[i] = smatrix.submatrix((0,0), (0,0)).shape[0]
        Ree[i] = smatrix.transmission((0,0), (0,0))
        Reh[i] = smatrix.transmission((0,1), (0,0))
        #G2[i] = smatrix.transmission((1,0), (0,0))
        print(str(i) + '/' + str(n-1))
    if plot:
        plotG(energies, N, Ree, Reh)
    return energies, N, Ree, Reh, N-Ree+Reh
def plotG(en, N, Ree, Reh, G):
    plt.ion()
    plt.figure()
    plt.plot(np.block([-np.flip(en,0), en]), np.block([np.flip(G,0), G]), label='G')
    plt.plot(np.block([-np.flip(en,0), en]), np.block([np.flip(N,0), N]), '-.', label='N')
    plt.plot(np.block([-np.flip(en,0), en]), np.block([np.flip(Reh,0), Reh]), '-.', label='Reh')
    plt.plot(np.block([-np.flip(en,0), en]), np.block([np.flip(Ree,0), Ree]), '-.', label='Ree')
    plt.legend()
    plt.xlabel('Bias in $\mu$eV')
    plt.ylabel('Conductance G')
    plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)+', $L_z=$'+str(Lz)+', $\Delta=$'+str(np.round(delta*1e3,1))+'meV, B1='+str(B1)+', W='+str(W*1e3)+'meV, t='+str(np.round(duration))+'s' )
    plt.tight_layout()
def plotLead(lead0, xlim=(0,np.pi/2), ylim=(-20,20), res=100, rand=True):
    plt.ion()
    plt.figure(figsize=(4,4))
    lead0 = lead0.finalized()
    bands = kw.physics.Bands(lead0)
    kz = np.linspace(xlim[0], xlim[1], res)
    energies = np.array([bands(k)*1000 for k in kz])
    n = energies.shape[1]
    if rand:
        energies += np.repeat(np.random.rand(n).reshape(n,1), res, axis=1).T * 1e-1
    plt.plot(kz, energies, linewidth=0.5)
    plt.ylim(ylim)
    plt.xlim((np.min(kz), np.max(kz)))
    if xdir != 0:
        plt.xlabel("$k_x a$")
    elif ydir != 0:
        plt.xlabel("$k_y a$")
    elif zdir != 0:
        plt.xlabel("$k_z a$")
    plt.ylabel("Energy in meV")
    plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)+', $L_z=$'+str(Lz)+', B='+str(Bx*1e3))
    plt.tight_layout()
    plt.savefig(next("C:/Users/Rafael/Desktop/MJ/transport/FT/figs/lead%s.pdf"))

    return energies
def save(filename, duration, en, N, Ree, Reh, G):
    f1 = open(filename, 'ab')

    pickle.dump(Lx, f1)
    pickle.dump(Ly, f1)
    pickle.dump(Lz, f1)
    pickle.dump(delta, f1)
    pickle.dump(W, f1)
    pickle.dump(duration, f1)
    pickle.dump(mu - (C0 - C1 * M0 / M1), f1) # mu relative to cone crossing
    pickle.dump(en, f1)
    pickle.dump(N, f1)
    pickle.dump(Ree, f1)
    pickle.dump(Reh, f1)
    pickle.dump(G, f1)
    pickle.dump(Bx, f1)
    pickle.dump(ggap, f1)
    pickle.dump(g, f1)
    pickle.dump(mu_local, f1)
    pickle.dump(B1, f1)
    pickle.dump(seed, f1)

    f1.close()
def sweep_disorder(max=50e-3, steps=10, energy=0.03e-3):
    ww = np.linspace(0e-3,max,steps)
    G = np.zeros(steps)
    N = np.zeros(steps)
    Reh = np.zeros(steps)
    Ree = np.zeros(steps)
    G = np.zeros(steps)
    global W #write changes of W to global variable

    for i in np.arange(len(ww)):
        W = ww[i]
        sys, lead0, lead1 = build_sys()
        en, N[i], Ree[i], Reh[i], G[i] = sim(sys, range=(energy,))
        print(i)

    filename = 'disorder/' + str(dt.datetime.now()).replace(':','_').replace('.','_').replace(' ','_')
    save(filename, en[0], ww, N, Ree, Reh, G)
def loop(rg):
    filename = str(dt.datetime.now()).replace(':','_').replace('.','_').replace(' ','_')+'.pickle'
    for i in np.arange(10):
        start(filename, rg)
def next(name):
    i = 0
    while os.path.exists(name % i):
        i += 1 
    return name % i
def nextrun(name):
    i = 10 
    while os.path.exists(name % i):
        i += 5 
    return i, name % i

def start(filename, rg):
    start_time = time.time()
    sys, lead0, lead1 = build_sys()
    en, N, Ree, Reh, G = sim(sys, range=rg, plot=False)
    duration = time.time()-start_time
    save(filename, duration, en, N, Ree, Reh, G)

if __name__ == '__main__':
    seed = int(sys.argv[1])
    np.random.seed(seed)
    rg = np.sort(np.block([np.linspace(0,130e-6,10)]))
    loop(rg)
    """
    start_time = time.time()
    sys, lead0, lead1 = build_sys()
    en, N, Ree, Reh, G = sim(sys, range=rg, plot=False)
    duration = time.time()-start_time
    plotG(en*1e6, N, Ree, Reh, G)
    plt.savefig(next("C:/Users/Rafael/Desktop/Dirac_NS_junction/_notes/fig_soc/cond%s.pdf"))
    #plt.savefig('Lz_%s.pdf' % Lz)
    #plt.close()
    #save(filename, duration, en, N, Ree, Reh, G)
    """
