from scipy.io import loadmat
import scipy.stats as sts
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from scikits import bootstrap
import seaborn as sns

n_folds = 2000

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

root_dir = "decoders/single_trials/erp/"


#### load shuffles
uncued_shuffles  = np.load(root_dir + "/shuffle_splits_processed/dec_uncued_shuffle_split.npy")
imp_shuffles  = np.load(root_dir + "/shuffle_splits_processed/dec_concat_shuffle_split_full.npy")
dec_shuffles  = np.load(root_dir + "/shuffle_splits_processed/dec_shuffle_split.npy")
all_shuffles = [dec_shuffles[:,:1000],uncued_shuffles[:,:1000], imp_shuffles[:,:1000]]

#### load folds

uncued_folds  = np.load(root_dir + "/folds/folds_uncued.npy")
imp_folds  = np.load(root_dir + "/folds/folds_impulse.npy")
dec_folds  = np.load(root_dir + "/folds/folds_unattended.npy")
folds = [dec_folds,uncued_folds, imp_folds]

#### load data
data = loadmat(root_dir + "early_late_smoothed_2factor_single_trials.mat")
dec_late1_sub = data['dec_late1_sub'][0]
dec_late2_sub = data['dec_late2_sub'][0]
dec_late= np.concatenate([dec_late1_sub,dec_late2_sub])

dec_uncued = loadmat(root_dir + "uncued_smoothed_2factor_single_trials.mat")
dec_uncued = dec_uncued["dec_uncued"][:,0]

dec_imp1 = loadmat(root_dir + "late_1_smoothed_2factor_single_trials_baselined.mat")['dec_late1_sub'][0]
dec_imp2 = loadmat(root_dir + "late_2_smoothed_2factor_single_trials_baselined.mat")['dec_late2_sub'][0]
dec_imp_baseline =  np.concatenate([dec_imp1,dec_imp2])


time_exp1 = np.linspace(-0.1,2.6460,1374)
time_exp2 = np.linspace(-0.1,1.1980,650)
time_imp = np.linspace(-0.1,1.6960,899)


times = [time_exp2, time_exp1,time_imp]

folds_done = True
if not folds_done:
    folds = []
    for i,data in enumerate([dec_late, dec_uncued, dec_imp_baseline]):
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

times[0] -= times[0][-1]
times[1] -= 2.1
times[2] -= 1.2

stim_on = [0.25+times[0][0]+0.1,0.25-2.1]
split_on = [0+times[0][0]+0.1,0-2.1]

# ploting
plt.figure(figsize=(10,5))
titles = ["unattended","discarded","reactivation"]
n_splits = [19,15,19]

y_max = [0.003, 0.007, 0.002]
for i in range(2):
    sorted_folds = folds[i]
    low_sorted_folds = np.mean(sorted_folds[:,:n_splits[i]],1)
    high_sorted_folds = np.mean(sorted_folds[:,-n_splits[i]:],1)

    plt.subplot(1,2,i+1)
    plt.title(titles[i])
    time = times[i]

    # p values against shuffles for high 
    shuffles = all_shuffles[i]
    m_h = np.mean(high_sorted_folds,0)
    s_h = np.mean(shuffles[0],1)

    p_h= np.array([np.mean(high_sorted_folds[:,t]<np.mean(s_h[:,t],0)) for t in range(len(time))])

    idx = p_h<0.05
    plt.plot(time[idx],y_max[i]*0.99*np.ones(sum(idx)),"|",color="black",ms=4)

    # sem
    ci_high = np.array([np.percentile(d,[16,100-16]) for d in high_sorted_folds.T])
    ci_low= np.array([np.percentile(d,[16,100-16]) for d in low_sorted_folds.T])

    fill_between(time,ci_high[:,0],ci_high[:,1],color='darkred',alpha=0.2)
    plot(time,np.mean(high_sorted_folds,0),color='darkred',label="high-decoding")

    fill_between(time,ci_low[:,0],ci_low[:,1],color='gray',alpha=0.2)
    plot(time,np.mean(low_sorted_folds,0),color='gray',label="low-decoding")

    plt.plot(time,np.mean(s_h,0),"k--",lw=0.5,color="darkred",label="shuffle predictor")
    plt.xlim(time[0],0)
    fill_between([stim_on[i],stim_on[i]+0.2],[-0.002,-0.002],[0.007,0.007],color='gray',alpha=0.2)
    fill_between([split_on[i],split_on[i]+0.25],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)

    plot(time,np.zeros_like(time),"k--")
    if i ==0: 
        plt.ylim(-0.0005,0.003)
        plt.yticks([0,0.0015,0.003])
        plt.ylabel("decoding strength")

    if i == 1:
        plt.ylim(-0.001,0.007)
        plt.yticks([0,0.0035,0.007])
        cue_on = 1.0500
        plt.fill_between([cue_on-2.1,cue_on+0.2-2.1],[-0.0002,-0.0002],[0,0],color="orange",alpha=0.5)
        plt.legend(frameon=False)

    ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    sns.despine()
    plt.xlabel("time from pinging (s)")

plt.savefig("figures/split_delay.svg")

## reactivation inset
figure(figsize=(2,2))

i = 2
sorted_folds = folds[i]
low_sorted_folds = np.mean(sorted_folds[:,:n_splits[i]],1)
high_sorted_folds = np.mean(sorted_folds[:,-n_splits[i]:],1)

time = times[i]

# p values against shuffles for high 
shuffles = all_shuffles[i]
m_h = np.mean(high_sorted_folds,0)
s_h = np.mean(shuffles[0],1)
p_h= np.array([np.mean(high_sorted_folds[:,t]<np.mean(s_h[:,t],0)) for t in range(len(time))])

idx = p_h<0.05
plt.plot(time[idx],y_max[i]*0.99*np.ones(sum(idx)),"|",color="black",ms=4)

# sem
ci_high = np.array([np.percentile(d,[16,100-16]) for d in high_sorted_folds.T])
ci_low= np.array([np.percentile(d,[16,100-16]) for d in low_sorted_folds.T])

# plotting
fill_between(time,ci_high[:,0],ci_high[:,1],color='darkred',alpha=0.2)
plot(time,np.mean(high_sorted_folds,0),color='darkred',label="high-decoding")

plt.plot(time,np.mean(s_h,0),"k--",lw=0.5,color="darkred")
fill_between(time,ci_low[:,0],ci_low[:,1],color='gray',alpha=0.2)
plot(time,np.mean(low_sorted_folds,0),color='gray',label="low-decoding")
plot(time,np.zeros_like(time),"k--")

plt.xlim(-0.1,0.5)
plt.xticks([0,0.25,0.5])
plt.ylim(-0.001,0.002)
plt.yticks([0,0.001,0.002])
ticklabel_format(style='sci', axis='y', scilimits=(0,0))
sns.despine()
# plt.xlabel("time from stimulus")
plt.savefig("figures/reactivation.svg")

times[0] = np.linspace(-0.1,1.1980,650)
times[1]= np.linspace(-0.1,2.6460,1374)

## delay p values
m_shuffle_1 = np.mean(all_shuffles[0][0][:,:,times[0]>0.25],-1)
m_delay_1 = np.mean(folds[0][:,-n_splits[0]:,times[0]>0.25],-1)


p_1_d = np.mean(np.mean(m_delay_1,-1) < np.mean(m_shuffle_1))

print("unattended, delay:  %f" % p_1_d)


m_shuffle_2 = np.mean(folds[1][:,-n_splits[1]:,times[1]>cue_on+0.2],-1)
m_delay_2 = np.mean(folds[1][:,-n_splits[1]:,times[1]>cue_on+0.2],-1)

p_2_d = np.mean(np.mean(m_delay_2,-1) < np.mean(m_shuffle_2))

print("uncueddelay:  %f" % p_2_d)

## pre-pinging p values


p_shuffle_1 = np.mean(all_shuffles[0][0][:,:,times[0] > times[0][-1]-0.25],-1)
m_ping_1 = np.mean(folds[0][:,-n_splits[0]:,times[0]> times[0][-1]-0.25],-1)

p_1 = np.mean(np.mean(m_ping_1,-1) < np.mean(p_shuffle_1))

print("unattended, ping:  %f" % p_1)

p_shuffle_2 = np.mean(folds[1][:,-n_splits[1]:,times[1]>times[1][-1]-0.25],-1)
m_ping_2 = np.mean(folds[1][:,-n_splits[1]:,times[1]>times[1][-1]-0.25],-1)

p_2 = np.mean(np.mean(m_ping_2,-1) < np.mean(p_shuffle_2))

print("uncued, ping:  %f" % p_2)


## pre pinging window

## not in the paper, found afterwards. It shows that there is a delay code if we use bigger windows than the one used in Fig 2. 

figure()
range_p = np.arange(0.25,1.1,0.01)
p_1_side_delay = [sts.ttest_1samp(np.concatenate([np.mean(d[:,times[0]>t],-1) for d in dec_late]),0)[1]/2 for t in range_p]

plt.plot(range_p-0.25,p_1_side_delay,"k")
plt.plot(range_p-0.25,np.ones_like(range_p)*0.05,"r")
plt.xlabel("window start (relative to beginning of delay)")
plt.title("1 sided t-test; window from x-axis until pinging")
plt.plot([times[0][-1]-0.2-0.25,times[0][-1]-0.2-0.25],[0,.15],"k--",alpha=0.5,label="window used in Barbosa et al")
plt.ylim([0,.15])
plt.xlim(range_p[0]-0.25,range_p[-1]-0.25)
plt.legend()
