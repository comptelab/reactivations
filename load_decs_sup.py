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


imp_on = 0.250+0.95

sns.set_style("ticks")
sns.set_context("talk", font_scale=1.5)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})

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
	erp_early,erp_late,alpha_early,alpha_late,beta_early,beta_late,theta_early,theta_late,broad_early,broad_late = data

	eearly = [gaussian_filter(erp, sigma=sig) for erp in erp_early[i]]
	elate = [gaussian_filter(erp, sigma=sig) for erp in erp_late[i]]

	aearly = [gaussian_filter(alpha, sigma=sig) for alpha in alpha_early[i]]
	alate = [gaussian_filter(alpha, sigma=sig) for alpha in alpha_late[i]]

	bearly = [gaussian_filter(beta, sigma=sig) for beta in beta_early[i]]
	blate = [gaussian_filter(beta, sigma=sig) for beta in beta_late[i]]

	tearly = [gaussian_filter(theta, sigma=sig) for theta in theta_early[i]]
	tlate = [gaussian_filter(theta, sigma=sig) for theta in theta_late[i]]

	bbearly = [gaussian_filter(broad, sigma=sig) for broad in broad_early[i]]
	bblate = [gaussian_filter(broad, sigma=sig) for broad in broad_late[i]]
	return [eearly,elate,aearly,alate,bearly,blate,tearly,tlate,bbearly,bblate]

alpha_late=loadmat("decoders/exp2_dec_mem_late_alpha.mat")["dec_mem_late"]
beta_late=loadmat("decoders/exp2_dec_mem_late_beta.mat")["dec_mem_late"]
theta_late=loadmat("decoders/exp2_dec_mem_late_theta.mat")["dec_mem_late"]
broad_late=loadmat("decoders/exp2_dec_mem_late_broadband.mat")["dec_mem_late"]

alpha_early=loadmat("decoders/exp2_dec_mem_early_alpha.mat")["dec_mem_early"]
beta_early=loadmat("decoders/exp2_dec_mem_early_beta.mat")["dec_mem_early"]
theta_early=loadmat("decoders/exp2_dec_mem_early_theta.mat")["dec_mem_early"]
broad_early=loadmat("decoders/exp2_dec_mem_early_broadband.mat")["dec_mem_early"]

erp_late=loadmat("decoders/exp2_dec_mem_late_erp.mat")["dec_mem_late"]
erp_early=loadmat("decoders/exp2_dec_mem_early_erp.mat")["dec_mem_early"]
time = loadmat('decoders/exp2_dec_mem_late_time.mat')["time"][0]

### smoothing and CI for EXP 2
data = erp_early,erp_late,alpha_early,alpha_late,beta_early,beta_late,theta_early,theta_late,broad_early,broad_late
idx=bootstrap.bootstrap_indexes(erp_late,n_samples=1000)
res = Parallel(n_jobs=num_cores)(delayed(smooth_i)(i,data) for i in idx)
mres = mean(res,2)


eearly,elate,aearly,alate,bearly,blate,tearly,tlate,bbearly,bblate = transpose(mres,[1,0,2])


ci_erp_early=array([percentile(erp,[2.5,97.5]) for erp in eearly.T])
ci_erp_late=array([percentile(erp,[2.5,97.5]) for erp in elate.T])

sem_erp_early=array([percentile(erp,[32/2,100-32/2]) for erp in eearly.T])
sem_erp_late=array([percentile(erp,[32/2,100-32/2]) for erp in elate.T])

ci_alpha_early=array([percentile(alpha,[2.5,97.5]) for alpha in aearly.T])
ci_alpha_late=array([percentile(alpha,[2.5,97.5]) for alpha in alate.T])

sem_alpha_early=array([percentile(alpha,[32/2,100-32/2]) for alpha in aearly.T])
sem_alpha_late=array([percentile(alpha,[32/2,100-32/2]) for alpha in alate.T])

ci_beta_early=array([percentile(beta,[2.5,97.5]) for beta in bearly.T])
ci_beta_late=array([percentile(beta,[2.5,97.5]) for beta in blate.T])

sem_beta_early=array([percentile(beta,[32/2,100-32/2]) for beta in bearly.T])
sem_beta_late=array([percentile(beta,[32/2,100-32/2]) for beta in blate.T])

ci_theta_early=array([percentile(theta,[2.5,97.5]) for theta in tearly.T])
ci_theta_late=array([percentile(theta,[2.5,97.5]) for theta in tlate.T])

sem_theta_early=array([percentile(theta,[32/2,100-32/2]) for theta in tearly.T])
sem_theta_late=array([percentile(theta,[32/2,100-32/2]) for theta in tlate.T])

ci_broad_early=array([percentile(broad,[2.5,97.5]) for broad in bbearly.T])
ci_broad_late=array([percentile(broad,[2.5,97.5]) for broad in bblate.T])

sem_broad_early=array([percentile(broad,[32/2,100-32/2]) for broad in bbearly.T])
sem_broad_late=array([percentile(broad,[32/2,100-32/2]) for broad in bblate.T])

plt.figure(figsize=(12,12))
plt.subplot(3,2,1)
plt.title("voltage")

plt.plot(time, mean(eearly,0),"skyblue")
plt.fill_between(time,sem_erp_early[:,0],sem_erp_early[:,1],alpha=0.3,label="attended",color="skyblue")
plt.plot(time,zeros(len(time)),"k--")


plt.plot(time, mean(elate,0),color="darkred")
plt.fill_between(time,sem_erp_late[:,0],sem_erp_late[:,1],alpha=0.15,label="unattended",color="darkred")
plt.plot(time,zeros(len(time)),"k--")


plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)

sig_bar(find(ci_erp_early[:,0]>0),time,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_erp_late[:,0]>0),time,[3e-3*0.97*0.975,3e-3*0.97],"darkred")

plt.xlim(-0.1,1.5)
plt.xlim(-.1,imp_on)
plt.ylim(-0.0005,0.002)
plt.yticks([0,0.001,0.002,0.003])
sns.despine()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
color_legend(["skyblue","darkred"],fontsize=20)

plt.subplot(3,2,3)
plt.title("broadband power")

plt.plot(time, mean(bbearly,0),color="skyblue")
plt.fill_between(time,sem_broad_early[:,0],sem_broad_early[:,1],alpha=0.3,color="skyblue")
plt.plot(time,zeros(len(time)),"k--")

plt.fill_between(time,sem_broad_late[:,0],sem_broad_late[:,1],alpha=0.15, color="darkred")
plt.plot(time, mean(bblate,0),color="darkred",label="unattended")
plt.plot(time,zeros(len(time)),"k--")


sig_bar(find(ci_broad_early[:,0]>0),time,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_broad_late[:,0]>0),time,[3e-3*0.97*0.975,3e-3*0.97],"darkred")


plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)


plt.xlim(-0.1,1.5)
plt.xlim(-.1,imp_on)
plt.ylim(-0.0005,0.003)
plt.yticks([0,0.001,0.002,0.003])
plt.ylabel("decoding strength")
sns.despine()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel("time from stimulus (s)")
	

plt.subplot(3,2,2)

plt.title(r"$\theta$"" power")

plt.plot(time, mean(tearly,0),color="skyblue")
plt.fill_between(time,sem_theta_early[:,0],sem_theta_early[:,1],alpha=0.3,color="skyblue")
plt.plot(time,zeros(len(time)),"k--")

plt.fill_between(time,sem_theta_late[:,0],sem_theta_late[:,1],alpha=0.15, color="darkred")
plt.plot(time, mean(tlate,0),color="darkred",label="unattended")
plt.plot(time,zeros(len(time)),"k--")


sig_bar(find(ci_theta_early[:,0]>0),time,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_theta_late[:,0]>0),time,[3e-3*0.97*0.975,3e-3*0.97],"darkred")


plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)


plt.xlim(-0.1,1.5)
plt.xlim(-.1,imp_on)
plt.ylim(-0.0005,0.003)
plt.yticks([0,0.001,0.002,0.003],[])
sns.despine()
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	
plt.subplot(3,2,4)
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

plt.xlim(-0.1,1.5)
plt.xlim(-.1,imp_on)
plt.ylim(-0.0005,0.003)
plt.yticks([0,0.001,0.002,0.003],[])

sns.despine()
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	
plt.subplot(3,2,6)
plt.title(r"$\beta$"" power")

plt.plot(time, mean(bearly,0),color="skyblue")
plt.fill_between(time,sem_beta_early[:,0],sem_beta_early[:,1],alpha=0.3,color="skyblue")
plt.plot(time,zeros(len(time)),"k--")

plt.fill_between(time,sem_beta_late[:,0],sem_beta_late[:,1],alpha=0.15, color="darkred")
plt.plot(time, mean(blate,0),color="darkred",label="unattended")
plt.plot(time,zeros(len(time)),"k--")


sig_bar(find(ci_beta_early[:,0]>0),time,[3e-3*0.975,3e-3],"skyblue")
sig_bar(find(ci_beta_late[:,0]>0),time,[3e-3*0.97*0.975,3e-3*0.97],"darkred")


plt.fill_between([0,0.250],[-0.0001,-0.0001],[0,0],color="gray",alpha=0.5)


plt.xlim(-0.1,1.5)
plt.xlim(-.1,imp_on)
plt.ylim(-0.0005,0.003)
plt.yticks([0,0.001,0.002,0.003])
sns.despine()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel("time from stimulus (s)")
	
plt.tight_layout()

savefig("figures/sup_dec.svg")
savefig("figures/sup_dec.png",dpi=300)