import matplotlib as mpl
mpl.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob('*.pickle')

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

count = 1
duration = 0

en = np.zeros(l)
Reh = np.zeros(l)
Ree = np.zeros(l)
N = np.zeros(l)
G = np.zeros(l)
for f in files:
    reader = open(f,'rb')

    try:
        while True:
            Lx = pickle.load(reader)
            Ly = pickle.load(reader)
            Lz = pickle.load(reader)
            delta = pickle.load(reader)
            W = pickle.load(reader)
            duration += pickle.load(reader)
            mu = pickle.load(reader)

            en = pickle.load(reader)

            N += pickle.load(reader)
            Ree += pickle.load(reader)
            Reh += pickle.load(reader)
            G += pickle.load(reader)

            Bx = pickle.load(reader)
            gap = pickle.load(reader)
            g = pickle.load(reader)
            mu_local = pickle.load(reader)
            B1 = pickle.load(reader)
            seed = pickle.load(reader)

    except EOFError:
        reader.close()

en = en / count
Reh = Reh / count
Ree = Ree / count
N = N / count
G = G / count
duration /= count

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
plt.plot(x, y, label='0K')
temps = [20, 30, 40, 50, 200, 400]
for T in temps:
    plt.plot(x, broaden(T), label=str(T)+'K')
plt.xlabel('Bias in $\mu$eV')
plt.ylabel('Conductance <G> ('+str(count)+' realizations)')
plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)\
          +', W='+str(round(W*1000))+'meV, $\Delta=$'+str(np.round(delta*1e6,1))+'$\mu$eV, $\mu$='+str(mu*1000)+'meV, $B_1$='+str(B1)+', t='\
          +str(np.round(duration))+'s')
#plt.xlim((-20,20))
plt.savefig('fig_avg_cond_fold.pdf')

