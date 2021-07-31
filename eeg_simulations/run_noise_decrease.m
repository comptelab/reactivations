clear all, close all, clc

rng(144)

cd C:\Users\diego.lozano\CODE\reactivations\eeg_simulations
% fieldtrip version: 'c4ce2f8f3'
% go to https://github.com/fieldtrip/fieldtrip/tree/c4ce2f8f3b1661d2f5520d656435699d358669e1
addpath 'C:\Users\diego.lozano\CODE\fieldtrip'
ft_defaults

fsample = 500;
t = -0.21:1/fsample:1.6; % time-axes

t0_imp     = 1.35; % (secs) time onset of the probe stimulus
t0_del     = 0.1; % (secs) jitter of the memory delay onset
t0_imp_avg = [1.3 1.5]; %(secs) time interval to compute decoding average

% get time samples
t0idx_imp     = nearest(t,t0_imp);
t0idx_del     = nearest(t,t0_del);
t0idx_imp_avg = nearest(t,t0_imp_avg);

var_decrease = 0.525;
noise_level  = 1;
ntrials = 200;

% time constants of exponentials
s_factor = 1; % standard deviation of the Gaussian kernel (see gsmooth.m)
erp = [];

for k=1:ntrials
  % update parameters that jitter across trials
  % time onsets
  t0idx_j     = round(normrnd(t0idx_del,3));
  t0idx_imp_j = round(normrnd(t0idx_imp,3));
  
  % A and B parameters that shape a Gamma PDF (see gampdf.m). The Gamma PDF
  % is used to emulate a potential working memory delay ERP component
  a_g1 = normrnd( 2,0.2);
  a_g2 = normrnd( 3,0.2);
  b_g1 = normrnd(130,0.5);
  b_g2 = normrnd(80,0.5);
  g1 = gamwaveform(1:size(t,2)-t0idx_del,a_g1,b_g1);
  g2 = gamwaveform(1:size(t,2)-t0idx_del,a_g2,b_g2);
  g1 = [zeros(1,t0idx_del) g1];
  g2 = [zeros(1,t0idx_del) g2];
  
  % delay-specific amplitude differences
  slow1 = g1.*0.5;
  slow2 = g2.*0.25;
  
  % trial-by-trial jitter of the noise quench.
  var_imp_j = round(normrnd(0.02*fsample,3));
  
  % simulate sample and probe stimuli
  t_sample = 1:size(t,2)>t0idx_j;
  t_probe  = 1:size(t,2)>t0idx_imp_j;
  
  pre_zeropad = size(find(t_probe==0),2);
  post_zeropad = size(t,2)-pre_zeropad;
  lst = find(t_sample==1);
  lst = lst(1);
  
  % noise with decrease of variance within-trial
  n1 = rand()*noise_level;
  n2 = rand()*var_decrease;
  
  t0idx_imp_avg_j = t0idx_imp_avg - var_imp_j;
  
  noise1 = [normrnd(0,0.75*n1,[1,t0idx_imp_avg_j(1)]) normrnd(0,0.75*n2,[size(t0idx_imp_avg_j(1)+1:t0idx_imp_avg_j(2))]) normrnd(0,0.75*n1,[size(t0idx_imp_avg_j(2)+1:size(t,2))])];
  noise2 = [normrnd(0,0.75*n1,[1,t0idx_imp_avg_j(1)]) normrnd(0,0.75*n2,[size(t0idx_imp_avg_j(1)+1:t0idx_imp_avg_j(2))]) normrnd(0,0.75*n1,[size(t0idx_imp_avg_j(2)+1:size(t,2))])];
  erp(k,1,:) = slow1 + noise1;
  erp(k,2,:) = slow2 + noise2;
end

%% dipole simulation
ftdir = fileparts(which('ft_defaults'));
mri = ft_read_mri(fullfile(ftdir,'template','headmodel','standard_mri.mat'));

% used later to plot dipoles
cfg = [];
cfg.resolution = 1;
cfg.xrange = [-100 100];
cfg.yrange = [-140 140];
cfg.zrange = [-80 120];
mris = ft_volumereslice(cfg, mri);
mris = ft_convert_units(mris, 'cm');

% align electrodes
label={'O2';'O1';'Oz';'Pz';'P4';'CP4';'P8';'C4';'TP8';'T8';'P7';'P3';...
  'CP3';'CPz';'Cz';'FC4';'FT8';'TP7';'C3';'FCz';'Fz';'F4';'F8';'T7';...
  'FT7';'FC3';'F3';'Fp2';'F7';'Fp1';'PO3';'P1';'POz';'P2';'PO4';...
  'CP2';'P6';'CP6';'C6';'PO8';'PO7';'P5';'CP5';'CP1';'C1';'C2';'FC2';...
  'FC6';'C5';'FC1';'F2';'F6';'FC5';'F1';'AF4';'AF8';'F5';'AF7';'AF3';...
  'Fpz';'AFz'};

elec = ft_read_sens(fullfile(ftdir,'template','electrode','standard_1005.elc'));

[sel1,sel2] = match_str(label,elec.label);
% isequal(elec.label(sel1),label(sel2))

chanpos = elec.chanpos(sel2,:);
elecpos = elec.elecpos(sel2,:);
chantype = elec.chantype(sel2,:);
chanunit = elec.chanunit(sel2,:);
elec.chantype = chantype;
elec.chanunit = chanunit;
elec.chanpos = chanpos;
elec.elecpos = elecpos;
elec.label = label;

vol = ft_read_vol(fullfile(ftdir,'template','headmodel','standard_bem.mat'));
vol = ft_convert_units(vol,'cm');

cfg                 = [];
cfg.elec            = ft_convert_units(elec,'cm');
cfg.headmodel       = ft_convert_units(vol,'cm');
cfg.reducerank      = 3;
cfg.channel         = 'all';
cfg.grid.resolution = 1;   % use a 3-D grid with a 1 cm resolution
cfg.grid.unit       = 'cm';
grid = ft_prepare_leadfield(cfg);

figure;
hold on;
subplot(221);ft_plot_vol(vol, 'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
subplot(221);ft_plot_mesh(grid.pos(grid.inside,:));

%%
pos1 = [-1.5 -8.6 1.5; 1.5 -8.6 1.5];
mom1 = [-1 1; -1 -1; -1 -1];

subplot(222);ft_plot_slice(mris.anatomy, 'transform', mris.transform, 'location',  [-2.1   -6.4    -1.4], 'orientation', [0 1 0], 'resolution', 0.1);
subplot(222);ft_plot_slice(mris.anatomy, 'transform', mris.transform, 'location',  [-2.1   -6.4    -1.4], 'orientation', [1 0 0], 'resolution', 0.1);
subplot(222);ft_plot_slice(mris.anatomy, 'transform', mris.transform, 'location',  [-2.1   -6.4    -1.4], 'orientation', [0 0 1], 'resolution', 0.1);
subplot(222);ft_plot_dipole(pos1,mom1,'color',[1 1 1]);
view([90 90]);
axis tight;
axis off;

%%
for k=1:size(erp,1)
  smt1 = gsmooth(squeeze(erp(k,1,:))',s_factor);
  smt2 = gsmooth(squeeze(erp(k,2,:))',s_factor);
  sig1{k}(1:2,:) = repmat(smt1,[2 1]);
  sig2{k}(1:2,:) = repmat(smt2,[2 1]);
end

cfg               = [];
cfg.dip.signal    = sig1;
cfg.dip.pos       = pos1;
cfg.dip.mom       = mom1;
cfg.fsample       = 600;
cfg.headmodel     = ft_convert_units(vol,'cm');
cfg.elec          = ft_convert_units(elec,'cm');
raw1 = ft_dipolesimulation(cfg);
x1 = ft_timelockanalysis([],raw1);
x1.time = t;
x1.elec = ft_convert_units(elec,'cm');

cfg.dip.signal    = sig2;
raw2 = ft_dipolesimulation(cfg);
x2 = ft_timelockanalysis([],raw2);
x2.time = t;
x2.elec = ft_convert_units(elec,'cm');

cfg = [];
cfg.channel = 'Oz';
cfg.figure = 'gcf';
subplot(223);
ft_singleplotER(cfg,x1,x2);


cfg                   = [];
cfg.xlim              = [0.25 1];
cfg.zlim              = [-0.003 0.003];
cfg.marker            = 'on';
cfg.markersymbol      = '.';
cfg.markersize        = 4;
cfg.highlight         = 'on';
cfg.highlightchannel  = 'Oz';
cfg.highlightsymbol   = '.';
cfg.highlightcolor    = [1 1 1];
cfg.highlightsize     = 12;
cfg.comment           = 'no';
cfg.interactive       = 'no';
cfg.figure = 'gcf';
subplot(224);ft_topoplotER(cfg,x1);colorbar;


%% mahalanobis distance
cfg = [];
cfg.keeptrials = 'yes';
cfg.channel = {'Pz','POz','Oz','P1','PO3','O1','P7','PO7','P5','P3','P2','PO4','O2','P8','PO8','P4','P6'};
x1_chan = ft_timelockanalysis(cfg,raw1);
x1_chan.time = t;
x2_chan = ft_timelockanalysis(cfg,raw2);
x2_chan.time = t;

angspace=(-pi:pi:pi)'; % angular space for tuning curve (in radians)
angspace(end)=[];
bin_width=pi; % width of each angle bin of the tuning curve (in radians)

mem_angles = [ones(size(x1_chan.trial,1),1).*pi; ones(size(x1_chan.trial,1),1).*2*pi];

xc = x1_chan;
xc.trial      = cat(1,x1_chan.trial,x2_chan.trial);
xc.sampleinfo = cat(1,x1_chan.sampleinfo,x2_chan.sampleinfo);
xc.trialinfo  = [zeros(size(x1_chan.trial,1),1); ones(size(x2_chan.trial,1),1)];
xc.avg        = squeeze(mean(xc.trial,1));
xc.dof        = ones(size(xc.avg)).*size(xc.trial,1);

cfg = [];
cfg.latency = [-0.2 1.6];
xc = ft_selectdata(cfg,xc);

slide_wins = [-0.2:0.2:1.6];
dec = zeros(size(slide_wins,2)-1,size(xc.trial,1),size(xc.trial,3));
eeg = zeros(size(slide_wins,2)-1,size(xc.trial,1),size(xc.trial,3));
oz_chan = match_str(xc.label,'Oz');

for win = 1:size(slide_wins,2)-1
  cfg = [];
  cfg.baseline = [slide_wins(win) slide_wins(win+1)];
  xc_b  = ft_timelockbaseline(cfg,xc);
  % select only Oz
  eeg(win,:,:) = squeeze(xc_b.trial(:,oz_chan,:));
  xc_b.trial= bsxfun(@minus, xc_b.trial, mean(xc_b.trial,2));
    
  dec(win,:,:) = mahalTune_func(xc_b.trial,mem_angles,angspace,bin_width);
  
  for k=1:size(dec(win,:,:),2)
    dec_s(win,k,:)=gsmooth(squeeze(dec(win,k,:)),s_factor)';
  end
end

trl1 = find(xc_b.trialinfo==0);
trl2 = find(xc_b.trialinfo==1);
for win = 1:size(slide_wins,2)-1
  tic;
  ci_eeg(win,1,:,:) = bootci(5000,@mean,squeeze(eeg(win,trl1,:)));
  ci_eeg(win,2,:,:) = bootci(5000,@mean,squeeze(eeg(win,trl2,:)));
  ci_dec(win,1,:,:) = bootci(5000,@mean,squeeze(dec(win,:,:)));
  ci_dec_s(win,1,:,:) = bootci(5000,@mean,squeeze(dec_s(win,:,:)));
  toc;
end

%% plot erp and decoding for each baseline
figure;
idx=reshape(1:18,2,9)';
t=-0.2:1/500:1.6;

cond=[1,2];
k=0;
for win = [1 4 7]
  k=k+1;
  hold all;
  subplot(3,2,k);shaded_ci(t,[squeeze(ci_eeg(win,cond(1),1,:))';squeeze(mean(eeg(win,trl1,:),2))';squeeze(ci_eeg(win,cond(1),2,:))'],'b',0.6,1);
  xlim([-0.2 1.6]);ylim([-0.015 0.025]);
  subplot(3,2,k);shaded_ci(t,[squeeze(ci_eeg(win,cond(2),1,:))';squeeze(mean(eeg(win,trl2,:),2))';squeeze(ci_eeg(win,cond(2),2,:))'],'r',0.6,1);
  xlim([-0.2 1.6]);ylim([-0.015 0.025]);
  subplot(3,2,k);fill([slide_wins(win),slide_wins(win),slide_wins(win+1),slide_wins(win+1)],[-0.015,-0.012,-0.012,-0.015],[65,212,26]./255,'EdgeColor','k')
  subplot(3,2,k);plot(t,zeros(size(t)),'k');
  k=k+1;  
  subplot(3,2,k);shaded_ci(t,[squeeze(ci_dec(win,1,1,:))';      squeeze(mean(dec(win,:,:),2))';   squeeze(ci_dec(win,1,2,:))'],[0.5 0.5 0.5],0.6,1);
  xlim([-0.2 1.6]);ylim([-0.15 0.43]);
  subplot(3,2,k);plot(t,zeros(size(t)),'k');
  subplot(3,2,k);shaded_ci(t(t0idx_imp_avg(1):t0idx_imp_avg(2)),[squeeze(ci_dec(win,1,1,t0idx_imp_avg(1):t0idx_imp_avg(2)))';      squeeze(mean(dec(win,:,t0idx_imp_avg(1):t0idx_imp_avg(2)),2))';   squeeze(ci_dec(win,1,2,t0idx_imp_avg(1):t0idx_imp_avg(2)))'],'k',0.6,1);
end


%% slide decoding of mahalanobis
wins = -0.2:0.2:1.6;
t0idx_imp_avg = nearest(t,[t0_imp_avg(1) t0_imp_avg(2)]);
ci=[];
for w=1:size(wins,2)-1
  dd = squeeze(mean(dec(w,:,t0idx_imp_avg(1):t0idx_imp_avg(2)),3))';
  ci(:,w) = bootci(5000,@mean,dd);
  bln_oi = nearest(t,[wins(w) wins(w+1)]);
  xtick{w} = [num2str(wins(w)) '-' num2str(wins(w+1)) 'ms'];
  probe(:,w) = mean(dd);
end
figure('position',[607   246   356   420]);
subplot(211);
imagesc(t,1:size(wins,2)-1,squeeze(mean(dec,2)),[0 0.25]);colorbar;colormap hot;
set(gca,'YTick',1:size(wins,2)-1);
set(gca,'YTickLabel',xtick,'FontSize',12)
hold all;
xlim([0.395 1.6]);
ylim([3.5 9.5]);
for w=4:size(wins,2)-1
  subplot(211);drawrect(slide_wins(w),slide_wins(w+1),0.5+(w-1),1.5+(w-1),'-',2,[65,212,26]./255);
  subplot(211);drawrect(t0_imp_avg(1),t0_imp_avg(2),0.5+(w-1),1.5+(w-1),'-',2,[0,0,255]./255);
end

colors = jet(size(1:size(wins,2)-1,2));
subplot(212);
x=0;
for j=fliplr(4:size(wins,2)-1)
  x=x+1;
  hold all
  b=barh(x,mean(probe(:,j),1)','EdgeColor', 'none', 'BarWidth', 0.4,'Facecolor',colors(x,:));
end
hold all;
ploterr(probe(fliplr(4:size(wins,2)-1)), (1:x),{ci(1,fliplr(4:size(wins,2)-1)),ci(2,fliplr(4:size(wins,2)-1))},{ones(1,x),ones(1,x)}, 'k.', 'abshhxy', 0);
xlim([-0.01 0.25]);
ylim([0 7]);
set(gca,'box','off');
xlabel('Decoding accuracy','FontSize',12)
set(gca,'FontSize',12)
set(gca,'YTick',1:6);
set(gca,'YTickLabel',xtick(fliplr(4:size(wins,2)-1)),'FontSize',12);
colorbar;
