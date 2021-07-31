
from __future__ import division
from matplotlib.pylab import *
import scipy.io as io
import glob
import sys
from scipy.stats import *
import seaborn as sns
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
import scipy.stats as stats


sns.set_style("ticks")
sns.set_context("talk", font_scale=0.8)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})

root_dir = "decoders/single_trials/erp/"

nboots = 5000

#### EXPERIMENT 1
### load data
imp_i = 1125
time_exp1 = np.linspace(-.1,2.646,1374)

data = loadmat(root_dir + "cued_uncued_smoothed_2factor_single_trials.mat")
dec_cued = data["dec_cued"][:,0]
dec_uncued = data["dec_uncued"][:,0]

all_uncued= []
all_cued= []


for i,d in enumerate(dec_uncued):

    #if i == 0: continue ## missing alpha decoder for subject #1 - correct this

    # load ERP decoders
    s = np.mean(d[:,imp_i-100:imp_i],-1)
    all_uncued.append(s)

    d = dec_cued[i]
    s = np.mean(d[:,imp_i-100:imp_i],-1)
    all_cued.append(s)

all_uncued = np.array(all_uncued)
all_cued = np.array(all_cued)


### EXPERIMENT 2
### load data
time = np.linspace(-0.1,1.1980,650)
imp_i = 650


data = loadmat(root_dir + "early_late_smoothed_2factor_single_trials.mat")
dec_early1_sub = data['dec_early1_sub'][0]
dec_early2_sub = data['dec_early2_sub'][0]
dec_late1_sub = data['dec_late1_sub'][0]
dec_late2_sub = data['dec_late2_sub'][0]

all_early = []
all_late= []

for i,d in enumerate(dec_early2_sub):
    #if i == 0: continue ## missing alpha decoder for subject #1 - correct this
    # load ERP decoders
    s1= np.mean(dec_early1_sub[i][:,imp_i-100:imp_i],1)
    s2 = np.mean(dec_early2_sub[i][:,imp_i-100:imp_i],1)
    all_early.append(s1); all_early.append(s2)

    s1= np.mean(dec_late1_sub[i][:,imp_i-100:imp_i],1)
    s2 = np.mean(dec_late2_sub[i][:,imp_i-100:imp_i],1)
    all_late.append(s1); all_late.append(s2)

all_early = np.array(all_early)
all_late = np.array(all_late)

### resampling
all_decs = [[all_early,all_late],[all_cued,all_uncued]]

Ts = []
Ps = []


subjects_range = []
trials_range = []

for i in range(2):

    decs = all_decs[i]

    subjects = np.arange(1,len(decs[0]))
    max_trials =max([len(a) for a in decs[0]])
    trials = np.arange(5,max_trials,max_trials//30)

    ## added twice for easier plotting
    subjects_range.append(subjects); subjects_range.append(subjects)
    trials_range.append(trials); trials_range.append(trials)

    idx_subjects = list(range(subjects[-1]))
    T = np.zeros((2,len(trials),len(subjects)))

    for nt,n_trials in enumerate(trials):
        print(nt)
        for _ in range(nboots):
            np.random.shuffle(idx_subjects)
            for ns,n_subjects in enumerate(subjects):

        ##### erp sampling
                # use random set of subjects
                c1_subjects = decs[0][idx_subjects[:n_subjects]]
                c2_subjects = decs[1][idx_subjects[:n_subjects]]

                # shuffle across trials
                [np.random.shuffle(c) for c in c1_subjects]
                [np.random.shuffle(c) for c in c2_subjects]
                
                # for each session, select the minimum between the max number of trials and 
                # the session's number of trials. This makes sure we are including all trials in the analyses
                c1_trials = np.concatenate([l[:min(n_trials,len(l))]for l in c1_subjects])
                c2_trials = np.concatenate([l[:min(n_trials,len(l))] for l in c2_subjects])

                t = ttest_1samp(c1_trials,0)[0]
                T[0,nt,ns] += t/nboots
            
                t = ttest_1samp(c2_trials,0)[0]
                T[1,nt,ns] += t/nboots


    T[0] = gaussian_filter(T[0], sigma=(2, 2))
    T[1] = gaussian_filter(T[1], sigma=(2, 2))

    # convert t values to p values
    P = np.zeros_like(T)

    for nt,n_trials in enumerate(trials):
        for ns,n_subjects in enumerate(subjects):
            P[0,nt,ns]=stats.t.sf(np.abs(T[0,nt,ns]), df=n_subjects*n_trials)/2
            P[1,nt,ns]=stats.t.sf(np.abs(T[1,nt,ns]), df=n_subjects*n_trials)/2

    Ts.append(T[0]); Ts.append(T[1])
    Ps.append(P[0]); Ps.append(P[1])


### plotting
plt.figure(figsize=(11,2))

for i in range(4):

    subjects = subjects_range[i]
    trials = trials_range[i]

    T = Ts[i]
    P = Ps[i]

    plt.subplot(1,4,i+1)
    plt.imshow(T,aspect="auto", 
                vmin=-1.5, vmax=5,origin='lower',extent=[subjects[0],subjects[-1],trials[0],trials[-1]],interpolation="nearest",cmap=sns.color_palette("magma", as_cmap=True))
            
    cbar = plt.colorbar()           
    cbar.set_label('t-value', rotation=270)  
    # cbar.set_ticks([-1,0,1,2,3])
    contours = plt.contour(subjects, trials, P, 8, colors='white',linewidths=1)
    plt.clabel(contours, inline=True, fontsize=10)
    if i == 0: plt.ylabel("erp decoding\nnumber of trials")
    plt.tick_params(left = False,bottom = False)
    plt.yticks(range(trials[0],trials[-1],200))
