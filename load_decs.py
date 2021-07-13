from scipy.io import loadmat
from scipy.stats import sem
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from scikits import bootstrap
import seaborn as sns
from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()
from scipy.ndimage import gaussian_filter
#import pylustrator 
#pylustrator.start()

sns.set_style("ticks")
sns.set_context("talk", font_scale=1.5)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})

# concatenation of several period
# :650 -> first delay

# load exp1 decoders
alpha_cued=loadmat("decoders/exp1_dec_alpha2.mat")["data"]["dec_cued"][0][0]
erp_cued=loadmat("decoders/exp1_dec_erp2.mat")["data"]["dec_cued"][0][0]
alpha_uncued=loadmat("decoders/exp1_dec_alpha2.mat")["data"]["dec_uncued"][0][0]
erp_uncued=loadmat("decoders/exp1_dec_erp2.mat")["data"]["dec_uncued"][0][0]
time_exp1=loadmat("decoders/exp1_dec_erp2.mat")["data"]["time"][0][0][0]

# load exp2 decoders
alpha_late=loadmat("decoders/exp2_dec_mem_late_alpha.mat")["dec_mem_late"]
beta_late=loadmat("decoders/exp2_dec_mem_late_beta.mat")["dec_mem_late"]
theta_late=loadmat("decoders/exp2_dec_mem_late_theta.mat")["dec_mem_late"]
erp_late=loadmat("decoders/exp2_dec_mem_late_erp.mat")["dec_mem_late"]
alpha_early=loadmat("decoders/exp2_dec_mem_early_alpha.mat")["dec_mem_early"]
beta_early=loadmat("decoders/exp2_dec_mem_early_beta.mat")["dec_mem_early"]
theta_early=loadmat("decoders/exp2_dec_mem_early_theta.mat")["dec_mem_early"]
erp_early=loadmat("decoders/exp2_dec_mem_early_erp.mat")["dec_mem_early"]
time = loadmat('decoders/exp2_dec_mem_late_time.mat')["time"][0]


find = lambda x: where(x)[0]

def color_legend(colors,loc="best",ncol=1,fontsize=15):
	l=plt.legend(frameon=False, loc=loc, ncol=ncol,fontsize=fontsize)
	for i,text in enumerate(l.get_texts()):
		text.set_color(colors[i])

	for item in l.legendHandles:
	    item.set_visible(False)

def sig_bar(sigs,axis,y,color):
	w=diff(axis)[0]
	for s in sigs:
		beg =axis[s]
		end = beg+w
		plt.fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)

sig = 10
def smooth_i(i,data):
	erp_early,erp_late,alpha_early,alpha_late = data
	eearly = [gaussian_filter(erp, sigma=sig) for erp in erp_early[i]]
	elate = [gaussian_filter(erp, sigma=sig) for erp in erp_late[i]]
	aearly = [gaussian_filter(alpha, sigma=sig) for alpha in alpha_early[i]]
	alate = [gaussian_filter(alpha, sigma=sig) for alpha in alpha_late[i]]
	return [eearly,elate,aearly,alate]


### smoothing and CI for EXP 1
idx=bootstrap.bootstrap_indexes(alpha_cued,n_samples=1000)
data = erp_cued,erp_uncued,alpha_cued,alpha_uncued
res = Parallel(n_jobs=num_cores)(delayed(smooth_i)(i,data) for i in idx)
mres = mean(res,2)
ecued,euncued,acued,auncued = mres[:,0],mres[:,1],mres[:,2],mres[:,3]


ci_alpha_cued=array([percentile(alpha,[2.5,97.5]) for alpha in acued.T])
ci_alpha_uncued=array([percentile(alpha,[2.5,97.5]) for alpha in auncued.T])

ci_erp_cued=array([percentile(erp,[2.5,97.5]) for erp in ecued.T])
ci_erp_uncued=array([percentile(erp,[2.5,97.5]) for erp in euncued.T])


sem_alpha_cued=array([percentile(alpha,[32/2,100-32/2]) for alpha in acued.T])
sem_alpha_uncued=array([percentile(alpha,[32/2,100-32/2]) for alpha in auncued.T])

sem_erp_cued=array([percentile(erp,[32/2,100-32/2]) for erp in ecued.T])
sem_erp_uncued=array([percentile(erp,[32/2,100-32/2]) for erp in euncued.T])




### smoothing and CI for EXP 2
data = erp_early,erp_late,alpha_early,alpha_late
idx=bootstrap.bootstrap_indexes(erp_late,n_samples=1000)
res = Parallel(n_jobs=num_cores)(delayed(smooth_i)(i,data) for i in idx)
mres = mean(res,2)
eearly,elate,aearly,alate = mres[:,0],mres[:,1],mres[:,2],mres[:,3]


ci_alpha_early=array([percentile(alpha,[2.5,97.5]) for alpha in aearly.T])
ci_alpha_late=array([percentile(alpha,[2.5,97.5]) for alpha in alate.T])

ci_erp_early=array([percentile(erp,[2.5,97.5]) for erp in eearly.T])
ci_erp_late=array([percentile(erp,[2.5,97.5]) for erp in elate.T])


sem_alpha_early=array([percentile(alpha,[32/2,100-32/2]) for alpha in aearly.T])
sem_alpha_late=array([percentile(alpha,[32/2,100-32/2]) for alpha in alate.T])

sem_erp_early=array([percentile(erp,[32/2,100-32/2]) for erp in eearly.T])
sem_erp_late=array([percentile(erp,[32/2,100-32/2]) for erp in elate.T])



imp_on = 0.250+0.95
cue_on = 1.0500
cue_on_i = argmin(abs(time - cue_on))
imp_on_exp1= 2.1500
plt.figure(figsize=(12,12))

plt.subplot(2,2,2)
plt.title("voltage")

plt.plot(time_exp1, mean(ecued,0),"skyblue")
plt.fill_between(time_exp1,sem_erp_cued[:,0],sem_erp_cued[:,1],alpha=0.3,label="cued",color="skyblue")
plt.plot(time_exp1,zeros(len(time_exp1)),"k--")


plt.plot(time_exp1, mean(euncued,0),"--",color="darkred")
plt.plot(time_exp1[:cue_on_i], mean(euncued,0)[:cue_on_i],color="darkred")
plt.fill_between(time_exp1,sem_erp_uncued[:,0],sem_erp_uncued[:,1],alpha=0.15,label="uncued",color="darkred")
plt.plot(time_exp1,zeros(len(time_exp1)),"k--")
plt.fill_between([1.65,2.15],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)



plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)
#plt.fill_between([imp_on_exp1,imp_on_exp1+0.1],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)
plt.fill_between([cue_on,cue_on+0.2],[-0.0001,-0.0001],[0,0],color="orange",alpha=0.5)

#plt.text(imp_on_exp1-0.2,-0.0004, "impulse",color="black",fontsize=20)
plt.text(cue_on-0.075,-0.0004, "cue",color="orange",fontsize=20)

sig_bar(find(ci_erp_cued[:,0]>0),time_exp1,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_erp_uncued[:,0]>0),time_exp1,[3e-3*0.97*0.975,3e-3*0.97],"darkred")

plt.xlim(-0.1,2.5)

plt.xlim(-0.1,imp_on_exp1)
#plt.xticks([0. , 0.5, 1. , 1.5,2., 2.5],[])
#ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim(-0.0005,0.003)
plt.yticks([0,0.001,0.002,0.003],[])
#sns.despine()
color_legend(["skyblue","darkred"],fontsize=20)
#ylabel("decoding strength")



plt.subplot(2,2,4)
plt.title(r"$\alpha$"" power")

plt.plot(time_exp1, mean(acued,0),color="skyblue")
plt.fill_between(time_exp1,sem_alpha_cued[:,0],sem_alpha_cued[:,1],alpha=0.3,color="skyblue")
plt.plot(time_exp1,zeros(len(time_exp1)),"k--")

plt.fill_between(time_exp1,sem_alpha_uncued[:,0],sem_alpha_uncued[:,1],alpha=0.15, color="darkred")
plt.plot(time_exp1, mean(auncued,0),"--",color="darkred")
plt.plot(time_exp1[:cue_on_i], mean(auncued,0)[:cue_on_i],color="darkred",label="uncued")
plt.fill_between([1.65,2.15],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)



plt.plot(time_exp1,zeros(len(time_exp1)),"k--")

sig_bar(find(ci_alpha_cued[:,0]>0),time_exp1,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_alpha_uncued[:,0]>0),time_exp1,[3e-3*0.97*0.975,3e-3*0.97],"darkred")

plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)
#plt.fill_between([imp_on_exp1,imp_on_exp1+0.1],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)
plt.fill_between([cue_on,cue_on+0.2],[-0.0001,-0.0001],[0,0],color="orange",alpha=0.5)

plt.yticks([0,0.001,0.002,0.003],[])
#ylabel("decoding strength")
plt.xlim(-0.1,2.5)
plt.xlim(-0.1,imp_on_exp1)
plt.ylim(-0.0005,0.003)
#ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.xticks([0. , 0.5, 1. , 1.5,2., 2.5],["stim.", ".5","1","1.5", "2", "2.5"])
sns.despine()

plt.xlabel("time from stimulus (s)")
	


plt.subplot(2,2,1)
plt.title("voltage")

plt.plot(time, mean(eearly,0),"skyblue")
plt.fill_between(time,sem_erp_early[:,0],sem_erp_early[:,1],alpha=0.3,label="attended",color="skyblue")
plt.plot(time,zeros(len(time)),"k--")


plt.plot(time, mean(elate,0),color="darkred")
plt.fill_between(time,sem_erp_late[:,0],sem_erp_late[:,1],alpha=0.15,label="unattended",color="darkred")
plt.plot(time,zeros(len(time)),"k--")
plt.fill_between([0.7,1.2],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)


plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)
#plt.fill_between([imp_on,imp_on+0.1],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)
#plt.text(1,-0.0004, "impulse",color="black",fontsize=15)
sig_bar(find(ci_erp_early[:,0]>0),time,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_erp_late[:,0]>0),time,[3e-3*0.97*0.975,3e-3*0.97],"darkred")

plt.xlim(-0.1,1.5)
plt.xlim(-.1,imp_on)
#plt.xticks(arange(0,1.6,0.5),[])
plt.ylim(-0.0005,0.002)
plt.yticks([0,0.001,0.002,0.003])
sns.despine()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
color_legend(["skyblue","darkred"],fontsize=20)
plt.ylabel("decoding strength")

	
plt.subplot(2,2,3)
plt.title(r"$\alpha$"" power")

plt.plot(time, mean(aearly,0),color="skyblue")
plt.fill_between(time,sem_alpha_early[:,0],sem_alpha_early[:,1],alpha=0.3,color="skyblue")
plt.plot(time,zeros(len(time)),"k--")

plt.fill_between(time,sem_alpha_late[:,0],sem_alpha_late[:,1],alpha=0.15, color="darkred")
plt.plot(time, mean(alate,0),color="darkred",label="unattended")
plt.plot(time,zeros(len(time)),"k--")


sig_bar(find(ci_alpha_early[:,0]>0),time,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_alpha_late[:,0]>0),time,[3e-3*0.97*0.975,3e-3*0.97],"darkred")


plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)
#plt.fill_between([imp_on,imp_on+0.1],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)


plt.xlim(-0.1,1.5)
plt.xlim(-.1,imp_on)
plt.ylim(-0.0005,0.003)
plt.yticks([0,0.001,0.002,0.003])
#plt.xticks([0. , 0.5, 1. , 1.5],["stim.", ".5","1","1.5"])
sns.despine()
plt.ylabel("decoding strength")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel("time from stimulus (s)")
plt.fill_between([0.7,1.2],[-0.0001,-0.0001],[0,0],color="black",alpha=0.5)


plt.tight_layout()

savefig("figures/fig4.png",dpi=300)
savefig("figures/fig4.svg",dpi=300)
plt.show()

