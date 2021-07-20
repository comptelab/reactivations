function y = gamwaveform(t,a,b)
num = exp(a-1);
den = (b*(a-1))^(a-1);
frm = (t.^(a-1)).*exp(-t./b);
y = (num/den).*frm;