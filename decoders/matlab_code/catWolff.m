function [out, dc1, goodtrl, badtrl, outofsynch] = catWolff(pre,post,baseline)
% this function concatenates two epoched datasets from Wolff et al., 2017
% Nat Neurosci that have different offsets. 
%   Inputs:
%       pre      - data matrix, trials x channels x time points
%       post     - data matrix, trials x channels x time points
%       baseline - vector that represents the time points that will be used
%   to concatenate pre and post after DC offset is corrected. For example, 
%   if baseline is 1:50, it follows pre(:,:,end-49:end), post(:,:,1:50) to
%   get the DC offset between segments.
%
% Diego Lozano-Soldevilla IDITRAP 14-Sep-2018 08:58:55

if any(diff(baseline,1)<1);
    error('baseline has to contain increasing number elements');
end
if any(diff(baseline,2));
    error('baseline is not a vector with consecutive values');
end
sizpre =  size(pre);
sizpost = size(post);
outofsynch=[];
% the problem in Wolff dataset is that some random subset of trials are
% misaligned by one sample (might be due to numerical issues when calling
% fieldtrip nearest.m)
kg=0;
kb=0;
dc1=[];
bmax = max(baseline);
flag=[];
goodtrl=[];
badtrl=[];
out = zeros(sizpre(1),sizpre(2),sizpost(end)+sizpre(end)-bmax-1);
for j=1:sizpre(1);
    pre_b  = squeeze(pre(j,:,end-(bmax-1):end));
    post_b = squeeze(post(j,:,1:bmax));
    
    jump = pre_b - post_b;
    absjump2 = abs(diff(jump,1,2));
    if any(mean(absjump2,2)<eps*15);
        kg=kg+1;
        goodtrl(kg)=j;
        flag = 'zero';        
        dc1(j,:) = jump(:,1);
        lastmem = squeeze(pre(j,:,:));
        dc1bcue = squeeze(post(j,:,:))+repmat(dc1(j,:)',[1 sizpost(3)]);
        out(j,:,:) = [lastmem(:,2:end-(bmax)) dc1bcue];%I drop the first sample to match the data with the misaligned trials
    end
    
    jump = pre_b(:,1:end-2)-post_b(:,2:end-1);
    absjump2 = abs(diff(jump,1,2));
    if any(mean(absjump2,2)<eps*15);
        kb=kb+1;
        badtrl(kb)=j;
        flag = 'lead';
        dc1(j,:) = jump(:,1);
        lastmem = squeeze(pre(j,:,:));
        dc1bcue = squeeze(post(j,:,:))+repmat(dc1(j,:)',[1 sizpost(3)]);
        tend = size(lastmem,2);
        tbeg = tend - (bmax-1);
        out(j,:,:) = [lastmem(:,1:tbeg-1) dc1bcue(:,2:end)];%here I pad the first sample with the first value of lastmem
    end
    
    jump = pre_b(:,2:end-1)-post_b(:,1:end-2);
    absjump2 = abs(diff(jump,1,2));
    if any(mean(absjump2,2)<eps*15);
        kb=kb+1;
        badtrl(kb)=j;
        flag = 'lag';
        dc1(j,:) = jump(:,1);
        lastmem = squeeze(pre(j,:,:));
        tend = size(lastmem,2);
        tbeg = tend - (bmax-1);
        lastmem = squeeze(pre(j,:,:));
        dc1bcue = squeeze(post(j,:,:))+repmat(dc1(j,:)',[1 sizpost(3)]);
        out(j,:,:) = [lastmem(:,3:tend-(bmax-1)) dc1bcue(:,1:end)];
    end
    
    if isempty(flag);
%         error(['trial ' num2str(j)]);
        outofsynch = [outofsynch; j];
    end
    flag=[];
end
