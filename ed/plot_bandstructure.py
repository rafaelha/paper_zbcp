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



reader = open('plotdata.pickle', 'rb')
Lx = pickle.load(reader)
Ly = pickle.load(reader)
delta = pickle.load(reader)
Bx = pickle.load(reader)
gap = pickle.load(reader)
kz = pickle.load(reader)
spectrum = pickle.load(reader)
pos = pickle.load(reader)
vecs = pickle.load(reader)

reader.close()

spos = pos*0;
sspectrum = spectrum*0;
for i in np.arange(len(kz)-1):
    ind = np.argmax(np.abs(vecs[:,:,i].T.conj()@vecs[:,:,i+1]),0);
    sspectrum[:,i+1] = spectrum[ind,i+1];
    spos[:,i+1] = pos[ind,i+1];
    
az=20
ylim=(-10,10)
nn = spectrum.shape[0]
plt.ion()

fig, ax = plt.subplots(1,1, num="one")
fig.set_size_inches(10,10)

norm = plt.Normalize(np.min(pos), np.max(pos))

for i in np.arange(nn):
    y = sspectrum[i,:] * 1000
    y = np.append(y, np.flip(y, 0))

    points = np.array([np.append(kz*az, np.flip(-kz*az, 0)), y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cdata = np.append(spos[i,:-1], np.flip(spos[i,1:], 0))

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
plt.xlim((-np.max(np.abs(kz))*az, np.max(np.abs(kz))*az))

plt.savefig('new_code_bandstructure2.pdf')
plt.close()
