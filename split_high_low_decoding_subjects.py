from scipy.io import loadmat
import scipy.stats as sts
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from scikits import bootstrap
import seaborn as sns

n_folds = 1000

sns.set_style("ticks")
sns.set_context("talk", font_scale=0.8)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})

def get_fold(session,n):

    trial_idx = list(range(len(session)))
    n_fold = len(session) // 2 
    folds1,folds2 = [],[]

    for _ in range(n):
        np.random.shuffle(trial_idx)
        folds1.append(np.mean(session[trial_idx[:n_fold]],0))
        folds2.append(np.mean(session[trial_idx[n_fold:]],0))

    return folds1,folds2

#### load data
root_dir = "/Users/jbarbosa/Dropbox/Neuro/papers/Reactivation/Dynamic_hidden_states/scripts/decoders_smoothed_EEG/"

data = loadmat(root_dir + "early_late_smoothed_2factor_single_trials.mat")
dec_late1_sub = data['dec_late1_sub'][0]
dec_late2_sub = data['dec_late2_sub'][0]
dec_late= np.concatenate([dec_late1_sub,dec_late2_sub])
time_exp2 = np.linspace(-0.1,1.1980,650)

dec_uncued = loadmat(root_dir + "uncued_smoothed_2factor_single_trials.mat")
dec_uncued = dec_uncued["dec_uncued"][:,0]
time_exp1 = np.linspace(-0.1,2.6460,1374)

data = loadmat(root_dir + "late_impulse_smoothed_2factor.mat")
dec_imp1 = data['dec_imp1_late1'][:,0]
dec_imp2 = data['dec_imp1_late2'][:,0]
dec_imp =  np.concatenate([dec_imp1[range(1,19)],dec_imp2[range(1,19)]])
time_imp = np.linspace(-0.1,0.498,300)

times = [time_exp2, time_exp1]
folds = []
for i,data in enumerate([dec_late, dec_uncued]):
    time = times[i]

    # get N 50/50 folds
    print("folds for ",i)
    dec_folds = np.array([get_fold(d,n_folds) for d in data])
    folds_1,folds_2  = dec_folds[:,0,:,:],dec_folds[:,1,:,:]

    # split-point right after cue
    bests = np.mean(folds_1[:,:,(time>0.25) & (time<0.45)],-1)

    # sort by split-point
    sorted_folds = np.array([folds_2[np.argsort(b),i] for i,b in enumerate(bests.T)])
  
    folds.append(sorted_folds)


# ploting
plt.figure(figsize=(10,5))
titles = ["unattended","discarded"]
n_splits = [19,15]

for i,sorted_folds in enumerate(folds):

    low_sorted_folds = np.mean(sorted_folds[:,:n_splits[i]],1)
    high_sorted_folds = np.mean(sorted_folds[:,-n_splits[i]:],1)

    plt.subplot(1,2,i+1)
    plt.title(titles[i])
    time = times[i]

    # 1-sided 95 CI
    ci_high = np.array([np.percentile(d,[5,95]) for d in high_sorted_folds.T])
    ci_low= np.array([np.percentile(d,[5,95]) for d in low_sorted_folds.T])

    # plotting
    fill_between(time,ci_high[:,0],ci_high[:,1],color='darkred',alpha=0.2)
    plot(time,np.mean(high_sorted_folds,0),color='darkred',label="high-decoding")

    # fill_between(time,ci_low[:,0],ci_low[:,1],color='gray',alpha=0.2)
    plot(time,np.mean(low_sorted_folds,0),color='black',label="low-decoding")

    fill_between([0.25,0.45],[-0.002,-0.002],[0.007,0.007],color='gray',alpha=0.2,label="split period")
    plot(time,np.zeros_like(time),"k--")
    if i ==0: 
        plt.xlim(-0.1,1.2)
        plt.ylim(-0.0005,0.003)
        plt.yticks([0,0.0015,0.003])
        fill_between([0,.25],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)
        plt.ylabel("decoding strength")

    else: 
        plt.xlim(-0.1,2.1)
        plt.ylim(-0.001,0.007)
        plt.yticks([0,0.0035,0.007])
        fill_between([0,.25],[-0.0002,-0.0002],[0,0],color="gray",alpha=0.5)
        cue_on = 1.0500
        plt.fill_between([cue_on,cue_on+0.2],[-0.0002,-0.0002],[0,0],color="orange",alpha=0.5)
        plt.legend(frameon=False)


    ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    sns.despine()
    plt.xlabel("time from stimulus")

plt.savefig("figures/split_delay.svg")

# "reactivations" split not done with CV due to different number of trials
plt.figure(figsize=(2,2))

# discard subject #1, as it doesnt exist for the impulse decoder.. 
dec_late = np.concatenate([dec_late1_sub[range(1,19)],dec_late2_sub[range(1,19)]])
dec_late = np.array([np.mean(np.mean(d,0)[(time_exp2>0.25) & (time_exp2<0.45)]) for d in dec_late])

# sort by early delay
idx = argsort(dec_late)

# load impulse data
data = loadmat(root_dir + "late_impulse_smoothed_2factor.mat")
dec_imp1 = data['dec_imp1_late1'][:,0]
dec_imp2 = data['dec_imp1_late2'][:,0]
dec_imp =  np.concatenate([dec_imp1[range(1,19)],dec_imp2[range(1,19)]])

dec_imp = np.array([np.mean(d,0) for d in dec_imp])
time_imp = np.linspace(-0.1,0.498,300)

low_dec_imp = dec_imp[idx[:18]]
high_dec_imp = dec_imp[idx[18:]]

ci_low = np.array([bootstrap.ci(d,alpha=1-0.69,n_samples=1000) for d in low_dec_imp.T])
ci_high = np.array([bootstrap.ci(d,alpha=1-0.69,n_samples=1000) for d in high_dec_imp.T])

# plotting
fill_between(time_imp,ci_high[:,0],ci_high[:,1],color='darkred',alpha=0.2)
plot(time_imp,np.mean(high_dec_imp,0),color='darkred',label="high-decoding")

fill_between(time_imp,ci_low[:,0],ci_low[:,1],color='gray',alpha=0.2)
plot(time_imp,np.mean(low_dec_imp,0),color='black',label="low-decoding")
plot(time_imp,np.zeros_like(time_imp),"k--")
plt.fill_between([0,.1],[-0.0002,-0.0002],[0,0],color="black",alpha=0.5)


sns.despine()
plt.xlim(-0.1,0.5)
plt.xlabel("time from pinging")
plt.ylim(-0.001,0.002)
plt.yticks([0,0.001,0.002])
ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig("figures/split_reactiations.svg")
