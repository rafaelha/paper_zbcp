import matplotlib as mpl
#mpl.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib import rc
rc('text.latex', preamble=r'\usepackage{sfmath}')

files = glob.glob('*.pickle')
ind = np.argsort(np.array([int(files[i].replace(".pickle","")) for i in range(len(files))]))

files = np.array(files)[ind]
def get_size():
    r = open(files[0], 'rb')
    pickle.load(r)
    pickle.load(r)
    pickle.load(r)
    pickle.load(r)
    pickle.load(r)
    pickle.load(r)
    pickle.load(r)
    ss = pickle.load(r)
    r.close()
    return len(ss)

l = get_size()

count = 0
dcount = 0
duration = 0

en = np.zeros(l)
Reh = np.zeros(l)
Ree = np.zeros(l)
N = np.zeros(l)
G = np.zeros(l)

pdf = PdfPages('fig_separate.pdf')
for f in files:
    reader = open(f,'rb')

    try:
        Lx = pickle.load(reader)
        Ly = pickle.load(reader)
        Lz = pickle.load(reader)
        delta = pickle.load(reader)
        W = pickle.load(reader)
        duration += pickle.load(reader)
        mu = pickle.load(reader)

        en = pickle.load(reader)

        N2 = pickle.load(reader)
        N += N2
        Ree += pickle.load(reader)
        Reh += pickle.load(reader)
        G2 = pickle.load(reader)
        G += G2

        Bx = pickle.load(reader)
        gap = pickle.load(reader)
        g = pickle.load(reader)
        mu_local = pickle.load(reader)
        B1 = pickle.load(reader)
        seed = pickle.load(reader)

        dcount += 1
        print(dcount)
        plt.figure()
        cond = np.logical_and(en<=delta, en>=-delta)
        en2 = np.block([en, -en[cond]])  * 1e6
        ind = np.argsort(en2)
        en2 = en2[ind]
        G2 = np.block([G2,G2[cond]])[ind]
        N2 = np.block([N2,N2[cond]])[ind]
        plt.plot(en2, G2)
        plt.plot(en2, N2)
        plt.xlabel('Bias in $\mu$eV')
        plt.ylabel('Conductance <G> ('+str(count)+' realizations)')
        plt.title('k='+str(Lz))
        pdf.savefig()
        plt.close()

        reader.close()

    except EOFError:
        reader.close()

pdf.close()
en = en
Reh = Reh
Ree = Ree
N = N
G = G
duration /= dcount

cond = np.logical_and(en<=delta, en>=-delta)
en = np.block([en, -en[cond]])  * 1e6
ind = np.argsort(en)
en = en[ind]
G = np.block([G,G[cond]])[ind]



res = 10000
x = np.linspace(np.min(en), np.max(en), res)
y = np.interp(x, en, G)

k = 8.617333262145/1e2 #mueV/(mK) 

def broaden(T=20):
    sech = 1/(4*k*T)/np.cosh(x/(2*k*T))**2*(x[1]-x[0])
    yt = np.convolve(y, sech, 'same')
    return yt


print('Realizations: '+str(count))
print('duration='+str(duration)+'s')

plt.rcParams["image.cmap"] = "Set1"

plt.ion()
plt.figure()
plt.plot(x, y, label='0K', color='navy')
temps = np.array([21, 80, 140, 210])
colors = ['navy', 'blue', 'fuchsia', 'orangered', 'firebrick']
i = 1
for T in temps:
    plt.plot(x, broaden(T), label=str(T)+'mK', color=colors[i])
    i += 1
plt.xlabel('Bias in $\mu$eV')
plt.ylabel('Conductance <G> ('+str(count)+' realizations)')
plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)\
          +', W='+str(round(W*1000))+'meV, $\Delta=$'+str(np.round(delta*1e6,1))+'$\mu$eV, $\mu$='+str(mu*1000)+'meV, $B_1$='+str(B1)+', t='\
          +str(np.round(duration))+'s')

ax = plt.gca()
i = len(colors)-1
for T in np.flip(temps,0):
    ax.add_patch(Rectangle((-1.75*k*T, -1), 3.5*k*T,
                           np.max(y)*2,alpha=0.1,fc=colors[i],lw=None))
    i -= 1
ax.add_patch(Rectangle((-delta*1e6, -1), 2*delta*1e6, np.max(y)*2,alpha=0.05,fc='k',lw=None))
    #plt.axvline(x=1.75*k*T)


plt.xlim((-600,600))
plt.ylim((0,np.max(y)*1.07))
plt.legend(frameon=False)
plt.savefig('fig_avg_cond_fold.pdf')

