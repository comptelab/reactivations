
from __future__ import division
from matplotlib.pylab import *
import scipy.io as io
import glob
import sys
from scipy.stats import *
import seaborn as sns
from scikits import bootstrap
from scipy.signal import detrend
from scipy.io import loadmat
from scipy.stats import sem
from scipy import stats
from scipy.ndimage.filters import gaussian_filter

sns.set_style("ticks")
sns.set_context("talk", font_scale=0.8)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})


cmap = sns.color_palette("magma", as_cmap=True)
nboots = 250

#### EXPERIMENT 1

# wolff 2017 trial by trial analyses
root_dir = "../baseline_decoders/single_trials2017/exp1/"
root_dir = "../baseline_decoders/single_trials2017/exp1/alpha_"

time = np.linspace(-.1,2.646,1374)
imp_i = 1125

all_cued = []
all_uncued= []


for sub in range(1,30):
    print(sub)

    # load single-trial decoders
    files=glob.glob(root_dir+"cued_exp1_single%i.mat" % (sub))[0]
    dec_cued=loadmat(files)["cued_trials"]
    dec_cued = np.mean(dec_cued[:,875:imp_i],-1)
    all_cued.append(dec_cued)

    # load single-trial decoders
    files=glob.glob(root_dir+"uncued_exp1_single%i.mat" % (sub))[0]
    dec_uncued=loadmat(files)["uncued_trials"]
    dec_uncued =  np.mean(dec_uncued[:,875:imp_i],-1)
    all_uncued.append(dec_uncued)

all_cued = np.array(all_cued)
all_uncued = np.array(all_uncued)

subjects1 = np.arange(5,len(all_cued),1)
trials1 = np.arange(300,980,980//20)
T = np.zeros((2,len(trials1),len(subjects1)))


for nt,n_trials in enumerate(trials1):
    print(nt)
    for _ in range(nboots):
        np.random.shuffle(all_cued)
        np.random.shuffle(all_uncued)
        for ns,n_subjects in enumerate(subjects1):
            t = ttest_1samp([np.mean(l[:n_trials]) for l in all_cued[:n_subjects]],0)
            T[0,nt,ns] += t[0]/nboots

            t = ttest_1samp([np.mean(l[:n_trials]) for l in all_uncued[:n_subjects]],0)
            T[1,nt,ns] += t[0]/nboots

T_s = T.copy()

T[0] = gaussian_filter(T_s[0], sigma=(2, 2))
T[1] = gaussian_filter(T_s[1], sigma=(2, 2))


# convert t values to p values
P = np.zeros_like(T)
for nt,n_trials in enumerate(trials1):
    for ns,n_subjects in enumerate(subjects1):
        P[0,nt,ns]=stats.t.sf(abs(T[0,nt,ns]), df=n_subjects)
        P[1,nt,ns]=stats.t.sf(abs(T[1,nt,ns]), df=n_subjects)


T1 = T.copy()
P1 = P.copy()


### EXPERIMENT 2

# wolff 2017 trial by trial analyses
root_dir = "../baseline_decoders/single_trials2017/"
root_dir = "../baseline_decoders/single_trials2017/alpha_"

time = loadmat('../decoders/exp2_dec_mem_late_time.mat')["time"][0]

imp_on = 0.250+0.95

imp_i = 650

T= 6
idx_imp = range(150,400)
idx_imp=range(imp_i+20,899)

all_early = []
all_late= []

for sub in range(1,20):
    print(sub)
    for session in [1,2]:
        # load single-trial decoders
        files=glob.glob(root_dir+"early_d_%i_single_dec_%i.mat" % (session, sub))[0]
        dec_early=loadmat(files)["dec_early%i" % session]
        dec_early = np.mean(dec_early[:,400:imp_i],-1)
        all_early.append(dec_early)

        # load single-trial decoders
        files=glob.glob(root_dir+"late_d_%i_single_dec_%i.mat" % (session, sub))[0]
        dec_late=loadmat(files)["dec_late%i" % session]
        dec_late =  np.mean(dec_late[:,400:imp_i],-1)
        all_late.append(dec_late)

all_early = np.array(all_early)
all_late = np.array(all_late)

all_early_ori = all_early.copy()
all_late_ori = all_late.copy()


subjects = range(5,39,1)
trials = range(350,724,720//20)
T = np.zeros((2,len(trials),len(subjects)))

for nt,n_trials in enumerate(trials):
    print(nt)
    for _ in range(nboots):
        np.random.shuffle(all_early)
        np.random.shuffle(all_late)
        for ns,n_subjects in enumerate(subjects):
            t = ttest_1samp([np.mean(l[:n_trials]) for l in all_early[:n_subjects]],0)
            T[0,nt,ns] += t[0]/nboots

            t = ttest_1samp([np.mean(l[:n_trials]) for l in all_late[:n_subjects]],0)
            T[1,nt,ns] += t[0]/nboots

T_s = T.copy()

T[0] = gaussian_filter(T_s[0], sigma=(2, 2))
T[1] = gaussian_filter(T_s[1], sigma=(2, 2))

# convert t values to p values
P = np.zeros_like(T)
for nt,n_trials in enumerate(trials):
    for ns,n_subjects in enumerate(subjects):
        P[0,nt,ns]=stats.t.sf(np.abs(T[0,nt,ns]), df=n_subjects)
        P[1,nt,ns]=stats.t.sf(np.abs(T[1,nt,ns]), df=n_subjects)


plt.figure(figsize=(11,2))

plt.subplot(1,4,1)
plt.title("attended") # [0.7-1.2 s]
plt.imshow(T[0],aspect="auto", 
               vmin=-1.5, vmax=3.8,origin='lower',extent=[subjects[0],subjects[-1],trials[0],trials[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()           
cbar.set_label('t-value', rotation=270)  
cbar.set_ticks([-1,0,1,2,3])
contours = plt.contour(subjects, trials, P[0], 8, colors='white',linewidths=1)
plt.clabel(contours, inline=True, fontsize=10)
plt.xlabel("number of subjects")
plt.ylabel("alpha decoding\nnumber of trials")
plt.tick_params(left = False,bottom = False)
plt.yticks(range(400,800,100))

plt.subplot(1,4,2)
plt.title("unattended")
plt.imshow(T[1],aspect="auto", 
                vmin=-1.5, vmax=3.8,origin='lower',extent=[subjects[0],subjects[-1],trials[0],trials[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()           
cbar.set_label('t-value', rotation=270)  
cbar.set_ticks([0,1,2,3])
contours = plt.contour(subjects, trials, P[1], 8, colors='white',linewidths=1)
plt.clabel(contours, inline=True, fontsize=10)
plt.xlabel("number of subjects")
plt.tick_params(left = False,bottom = False)
plt.yticks([])


plt.subplot(1,4,3)
plt.title("cued") #[1.65-2.15 s]
plt.imshow(T1[0],aspect="auto", 
               vmin=-1.5, vmax=3.8,origin='lower',extent=[subjects1[0],subjects1[-1],trials1[0],trials1[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()           
cbar.set_label('t-value', rotation=270)  
cbar.set_ticks([-1,0,1,2,3])
contours = plt.contour(subjects1, trials1, P1[0], 8, colors='white',linewidths=1)
plt.clabel(contours, inline=True, fontsize=10)
plt.xlabel("number of subjects")
plt.tick_params(left = False,bottom = False)
plt.yticks(range(300,1000,200))
plt.xticks(range(5,30,10))

plt.subplot(1,4,4)
plt.title("uncued")
plt.imshow(T1[1],aspect="auto", 
                vmin=-1.5, vmax=3.8,origin='lower',extent=[subjects1[0],subjects1[-1],trials1[0],trials1[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()           
cbar.set_label('t-value', rotation=270)  
cbar.set_ticks([-1,0,1,2,3])
contours = plt.contour(subjects1, trials1, P1[1], [0.05], colors='black',linewidths=1)
contours = plt.contour(subjects1, trials1, P1[1], 8, colors='white',linewidths=1)
plt.clabel(contours, inline=True, fontsize=10)
plt.xlabel("number of subjects")
plt.tick_params(left = False,bottom = False)
plt.yticks([])
plt.xticks(range(5,30,10))

plt.savefig("figures/alpha_power_analyses.svg")
#plt.tight_layout()
plt.show()



all_early_combined = []
all_late_combined = []

for i in range(0,38,2):
    all_early_combined.append(list(all_early_ori[i]) + list(all_early_ori[i+1]))
    all_late_combined.append(list(all_late_ori[i]) + list(all_late_ori[i+1]))


early = np.array([np.mean(l) for l in all_early_combined])
late = np.array([np.mean(l) for l in all_late_combined])
cued = np.array([np.mean(l) for l in all_cued])
uncued = np.array([np.mean(l) for l in all_uncued])


plt.figure()
bar(0,np.mean(early),color="skyblue")
bar(1,np.mean(late),color="darkred")
plt.errorbar(0, np.mean(early), yerr=bootstrap.ci(early,alpha=1-0.69,output='errorbar'), 
        fmt='ko')
plt.errorbar(1, np.mean(late), yerr=bootstrap.ci(late,alpha=1-0.69,output='errorbar'), 
        fmt='ko')
axis("off")

plt.savefig("figures/alpha_early_late.svg")


plt.figure()
bar(0,np.mean(cued),color="skyblue")
bar(1,np.mean(uncued),color="darkred")
plt.errorbar(0, np.mean(cued), yerr=bootstrap.ci(cued,alpha=1-0.69,output='errorbar'), 
        fmt='ko')
plt.errorbar(1, np.mean(uncued), yerr=bootstrap.ci(uncued,alpha=1-0.69,output='errorbar'), 
        fmt='ko')
axis("off")
plt.savefig("figures/alpha_cue_uncued.svg")
