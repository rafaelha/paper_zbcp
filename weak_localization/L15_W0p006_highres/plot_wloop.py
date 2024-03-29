import matplotlib as mpl
mpl.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob('*.pickle')

W_list = []

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

for f in files:
    reader = open(f,'rb')

    try:
        while True:
            Lx = pickle.load(reader)
            Ly = pickle.load(reader)
            Lz = pickle.load(reader)
            delta = pickle.load(reader)
            W = pickle.load(reader)
            W_list.append(W)
            pickle.load(reader)
            pickle.load(reader)

            pickle.load(reader)

            pickle.load(reader)
            pickle.load(reader)
            pickle.load(reader)
            pickle.load(reader)

            pickle.load(reader)
            pickle.load(reader)
            pickle.load(reader)
            pickle.load(reader)
            pickle.load(reader)
            pickle.load(reader)

    except EOFError:
        reader.close()

W_list = list( dict.fromkeys(W_list) )

for W in W_list:
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
                    W2 = pickle.load(reader)
                    dur = pickle.load(reader)
                    mu = pickle.load(reader)

                    en2 = pickle.load(reader)

                    N2 = pickle.load(reader)
                    Ree2 = pickle.load(reader)
                    Reh2 = pickle.load(reader)
                    G2 = pickle.load(reader)

                    Bx = pickle.load(reader)
                    gap = pickle.load(reader)
                    g = pickle.load(reader)
                    mu_local = pickle.load(reader)
                    B1 = pickle.load(reader)
                    seed = pickle.load(reader)

                    if W2 == W:
                            count += 1
                            en += en2
                            N += N2
                            Ree += Ree2
                            Reh += Reh2
                            G += G2
                            duration += dur
            except EOFError:
                reader.close()

        en = en / count
        Reh = Reh / count
        Ree = Ree / count
        N = N / count
        G = G / count
        duration /= count

        cond = np.logical_and(en<=delta, en>=-delta)
        en = np.block([en, -en[cond]]) * 1e6
        ind = np.argsort(en)
        en = en[ind]
        G = np.block([G,G[cond]])[ind]
        print('Realizations: '+str(count))
        print('duration='+str(duration/3600)+'h')
        plt.ion()
        plt.figure()
        plt.plot(en, G)
        plt.xlabel('Bias in $\mu$eV')
        plt.ylabel('Conductance <G> ('+str(count)+' realizations)')
        plt.title('$L_x=$'+str(Lx)+', $L_y=$'+str(Ly)+', $L_z=$'+str(Lz)\
                  +', W='+str(round(W*1000))+'meV, $\Delta=$'+str(np.round(delta*1e6,1))+'$\mu$eV, $\mu$='+str(mu*1000)+'meV, $B_1$='+str(B1)+', t='\
                  +str(np.round(duration))+'s')
        plt.savefig('fig_avg_cond_fold'+str(W)+'.pdf')

