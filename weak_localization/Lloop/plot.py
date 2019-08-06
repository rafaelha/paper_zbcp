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

count = 0
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

            enr = pickle.load(reader)
            en += enr

            N += pickle.load(reader)
            Ree += pickle.load(reader)
            Reh += pickle.load(reader)
            Gr = pickle.load(reader)
            G += Gr
            Bx = pickle.load(reader)
            gap = pickle.load(reader)
            g = pickle.load(reader)
            mu_local = pickle.load(reader)
            B1 = pickle.load(reader)
            seed = pickle.load(reader)


            plt.figure()
            plt.plot(enr, Gr, label='G')
            plt.xlabel('Bias in $\mu$eV')
            plt.ylabel('Conductance <G> ('+str(count)+' realizations)')
            plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)+', $L_z=$'+str(Lz)\
                            +', W='+str(round(W*1000))+'meV, $\Delta=$'+str(np.round(delta*1e6,1))+'$\mu$eV, $\mu$='+str(mu*1000)+'meV, $B_1$='+str(B1)+', t='\
                            +str(np.round(duration))+'s')
            plt.savefig('fig/fig_avg_cond_'+str(count)+'.pdf')
            plt.close()
            count += 1
    except EOFError:
        reader.close()

en = en / count * 1e6
Reh = Reh / count
Ree = Ree / count
N = N / count
G = G / count
duration /= count

print('Realizations: '+str(count))
print('duration='+str(duration)+'s')
plt.ion()
plt.figure()
plt.plot(en, G, label='G')
plt.plot(en, N, '-.', label='N')
plt.plot(en, Reh, '-.', label='Reh')
plt.plot(en, Ree, '-.', label='Ree')
plt.legend()
plt.xlabel('Bias in $\mu$eV')
plt.ylabel('Conductance <G> ('+str(count)+' realizations)')
plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)+', $L_z=$'+str(Lz)\
          +', W='+str(round(W*1000))+'meV, $\Delta=$'+str(np.round(delta*1e6,1))+'$\mu$eV, $\mu$='+str(mu*1000)+'meV, $B_1$='+str(B1)+', t='\
          +str(np.round(duration))+'s')
plt.savefig('fig_avg_cond.pdf')

