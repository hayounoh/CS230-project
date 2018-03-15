clear
clc

f=600;
Fs=6000;
t=0:1/Fs:0.3;
n=0:1:length(t);
x=cos(2*pi*(400/Fs)*n)+2*sin(2*pi*(1100/Fs)*n);
y=fft(x);
subplot(211)
plot(n,x)
title('f(t) vs. t')
xlabel('Time t')
ylabel('f(t)')

freqaxis=Fs*(linspace(-0.5,0.5, length(y)));
subplot(212)
plot(freqaxis,fftshift(abs(y)));
title('|F(s)| vs. s')
xlabel('Frequency s')
ylabel('Magnitude of F')