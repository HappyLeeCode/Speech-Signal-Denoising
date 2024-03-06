clear ALL; 
clc; 
clf

%% 录制人声并保存为SoundRecording.wav
%  如果要新录制人声，请注释掉第20行，然后再解除第8到16行的注释

% Fs = 8000;                 %采样率（默认8000）
% nBits = 16;                %采样位数
% nChannels = 2;             %通道数
% time = 15;                 %录取时长/ s
% sound1 = audiorecorder(Fs,nBits,nChannels);
% recordblocking(sound1,time);         %将音频录制到 audiorecorder 对象中
% Recording = getaudiodata(sound1); %将录制的音频信号存储
% audiowrite('./VoiceRecord.wav',Recording,Fs);
% [x,Fs]=audioread('./VoiceRecord.wav');        % 原始音频x 采样频率Fs


%% 读取音频原始信号
 [x,Fs]=audioread('./VoiceRecord1.wav');        % 原始音频x 采样频率Fs
% sound(x,Fs);                                  % 播放原始音频

% 如果要使用下面采集的别的录音，请注释掉前面的相关读取操作
% [x,Fs]=audioread('music.wav');               % 一段音乐录音（26s）
% [x,Fs]=audioread('VoiceRecord2.wav');        % 备用人声录音（10s)
% [x,Fs]=audioread('niganma.wav');             % 网络人声录音（4s）
% sound(x,Fs);                                 % 播放原始音频


N=length(x);                     %采样点数
n=0:N-1;  
t=n/Fs;                          %时域范围
Wx=2*n*pi/N;                     %频域范围

x=x(:,1);                        %双声道音频转单声道
X=fft(x);                        %快速傅里叶变换

% 绘制原始信号图形
figure(1);
subplot(2,1,1);
plot(t,x);
title('原始语音信号的时域波形图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(2,1,2);
plot(Wx/pi,abs(X)); 
title('原始语音信号采样后的频谱图');
xlabel("归一化频率/ \pi rad");
ylabel("幅值");


%% 加噪混频
%  内容：语音信号（低频） + 高斯噪声（高频）
%  自定义的Gnoisegen(x,snr)函数       (详细代码，查看Gnoisegen.m文件)

%  设置信噪比snr不同，即对应噪声强度不同  可设置为：10，1，0.1，0.01
snr=1;

[x_n,hpf_noise,noise] = Gnoisegen(x,Fs,snr); 
% 输入： x-原始语音信号       Fs-信号的采样频率        snr-信噪比 
% 输出： y-最终合成的带噪信号  hpf_noise-高频高斯噪声  noise-全通高斯噪声

noise_fft=fft(noise);
hpf_fft=fft(hpf_noise);
X_n=fft(x_n);

% 绘制噪声信号图形
figure(2);
subplot(4,1,1);
plot(t,noise);
title('高斯噪声时域图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(4,1,2);
plot(Wx/pi,abs(noise_fft));
title('高斯噪声频谱图');
xlabel("归一化频率/ \pi rad");
ylabel("幅值");
subplot(4,1,3);
plot(t,hpf_noise);
title('带限高斯噪声时域图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(4,1,4);
plot(Wx/pi,abs(hpf_fft));
title('带限高斯噪声频谱图');
xlabel("归一化频率/ \pi rad");
ylabel("幅值");

% 绘制加噪信号图形
figure(3);
subplot(2,1,1);
plot(t,x_n);
title('音频信号加带限噪声后的时域图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(2,1,2);
plot(Wx/pi,abs(X_n)); 
title('音频信号加带限噪声后的频谱图');
xlabel("归一化频率/\pi rad");
ylabel("幅值");


%% 去噪
%  内容：带噪信号+低通滤波器
%  3组IIR滤波器：巴特沃斯（脉冲响应不变法）、巴特沃斯（双线性变换法）、椭圆
%  3组FIR滤波器：矩形窗、哈明窗、凯塞窗

% IIR滤波器设计
Wp=0.4*pi;
Ws=0.5*pi;
Rp=1;
Rs=20;
T=1;
fs=1/T;
omegap=2*tan(Wp/2);
omegas=2*tan(Ws/2);

% 巴特沃斯——脉冲响应不变法
[N,Wn]=buttord(Wp,Ws,Rp,Rs,'s');
[B1,A1]=butter(N,Wn,"low",'s');
[Bz1,Az1]=impinvar(B1,A1,fs);
[H1,w1]=freqz(Bz1,Az1);
Hz1=20*log10(abs(H1));

% 巴特沃斯——双线性变换法
[N,Wn]=buttord(omegap,omegas,Rp,Rs,'s');
[B2,A2]=butter(N,Wn,"low",'s');
[Bz2,Az2]=bilinear(B2,A2,fs);
[H2,w2]=freqz(Bz2,Az2);
Hz2=20*log10(abs(H2));

% 椭圆滤波器——双线性变换法
[N,Wn]=ellipord(omegap,omegas,Rp,Rs,'s');
[B3,A3]=ellip(N,Rp,Rs,Wn,"low",'s');
[Bz3,Az3]=bilinear(B3,A3,fs);
[H3,w3]=freqz(Bz2,Az2);
Hz3=20*log10(abs(H3));

% 绘制IIR滤波器的幅频特性曲线
figure(4);
subplot(1,3,1);
plot(w1/pi,Hz1)
axis([0 2 -100 10]);
title("IIR1 巴特沃斯低通滤波器（脉冲响应不变法）");
xlabel("归一化频率/ \pi rad")
ylabel("Hz(dB)")
subplot(1,3,2);
plot(w2/pi,Hz2)
axis([0 2 -100 10]);
title("IIR2 巴特沃斯低通滤波器（双线性变换法）");
xlabel("归一化频率/ \pi rad")
ylabel("Hz(dB)")
subplot(1,3,3);
plot(w3/pi,Hz3)
axis([0 2 -100 10]);
title("IIR3 椭圆低通滤波器（双线性变换法）");
xlabel("归一化频率/ \pi rad")
ylabel("Hz(dB)")

% 使用IIR滤波器滤波
x_IIR1=filter(Bz1,Az1,x_n);
X_IIR1=fft(x_IIR1);
x_IIR2=filter(Bz2,Az2,x_n);
X_IIR2=fft(x_IIR2);
x_IIR3=filter(Bz3,Az3,x_n);
X_IIR3=fft(x_IIR3);

figure(5);
subplot(4,1,1);plot(Wx/pi,abs(X_n)); title('滤波前频谱');xlabel("频率/\pi rad");ylabel("幅值");
subplot(4,1,2);plot(Wx/pi,abs(X_IIR1)); title('巴特沃斯低通IIR滤波后的频谱（脉冲响应不变法）');xlabel("频率/ \pi rad");ylabel("幅值");
subplot(4,1,3);plot(Wx/pi,abs(X_IIR2)); title('巴特沃斯低通IIR滤波后的频谱（双线性变换法）');xlabel("频率/ \pi rad");ylabel("幅值");
subplot(4,1,4);plot(Wx/pi,abs(X_IIR3)); title('椭圆低通IIR滤波后的频谱（双线性变换法）');xlabel("频率/ \pi rad");ylabel("幅值");

% % FIR滤波器设计(固定滤波器阶数为10阶)
% 
% N=11;              %窗宽
% M=N-1;             %FIR滤波器阶数
% Wc=0.5*pi;         %截止频率
% % 矩形窗
% wind=boxcar(N);
% b1=fir1(M,Wc/pi,'low',wind);
% a1=1;
% [H1,w1]=freqz(b1,a1);
% % 哈明窗
% wind=hamming(N);
% b2=fir1(M,Wc/pi,'low',wind);
% a2=1;
% [H2,w2]=freqz(b2,a2);

% FIR滤波器设计(阶数由查表精确确定)
Wp=0.4*pi;
Ws=0.5*pi;
Wc=(Wp+Ws)/2;
Rs=20;
deltaw=Ws-Wp;

% 用来查表的确定窗宽N和过度带宽deltaw的关系
N1=ceil(1.8*pi/deltaw);  % 矩形窗
N2=ceil(6.6*pi/deltaw);  % 哈明窗
% FIR滤波器阶数
M1=N1-1;
M2=N2-1;

% 矩形窗
wind=boxcar(N1);
b1=fir1(M1,Wc/pi,'low',wind);
a1=1;
[H1,w1]=freqz(b1,a1);

% 哈明窗
wind=hamming(N2);
b2=fir1(M2,Wc/pi,'low',wind);
a2=1;
[H2,w2]=freqz(b2,a2);

% 凯塞窗
fcuts = [0.4  0.5]; %归一化频率omega/pi，这里指通带截止频率、阻带起始频率
mags = [1 0];
devs = [0.05 10^(-2.5)];
[n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs);  %计算出凯塞窗N，beta的值
b3 = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale'); 
a3=1;
[H3,w3]=freqz(b3,a3);

% FIR带通滤波器设计
firBP_N = 100;  % FIR滤波器阶数
firBP_cutoff_freq = [300, 3400];  % FIR滤波器截止频率
firBP_cutoff_freq_norm = firBP_cutoff_freq / Fs;  % 归一化截止频率
firBP_filter = fir1(firBP_N, firBP_cutoff_freq_norm,"bandpass");  % FIR滤波器设计
[H_BP,wBP]=freqz(firBP_filter,1);

% 绘制FIR低通滤波器的幅频特性曲线
figure(6);
subplot(3,1,1);
Hz1=20*log10(abs(H1));
plot(w1/pi,Hz1);
title("矩形窗设计的FIR低通滤波器");
xlabel("频率/ \pi rad")
ylabel("Hz(dB)")
subplot(3,1,2);
Hz2=20*log10(abs(H2));
plot(w2/pi,Hz2);
title("哈明窗设计的FIR低通滤波器");
xlabel("频率/ \pi rad");
ylabel("Hz(dB)");
subplot(3,1,3);
Hz3=20*log10(abs(H3));
plot(w3/pi,Hz3);
title("凯塞窗设计的FIR低通滤波器");
xlabel("频率/ \pi rad");
ylabel("Hz(dB)");

% 绘制FIR带通滤波器的幅频特性曲线
figure(7);
Hz_BP=20*log10(abs(H_BP));
plot(wBP/pi,Hz_BP);
title("哈明窗设计的FIR带通滤波器");
xlabel("频率/ \pi rad")
ylabel("Hz(dB)")

% 使用FIR滤波器
x_FIR1=filter(b1,a1,x_n);
X_FIR1=fft(x_FIR1);
x_FIR2=filter(b2,a2,x_n);
X_FIR2=fft(x_FIR2);
x_FIR3=filter(b3,a3,x_n);
X_FIR3=fft(x_FIR3);
x_BP =filter(firBP_filter, 1, x_n);
X_BP =fft(x_BP);


figure(8);
subplot(4,1,1);plot(Wx/pi,abs(X_n)); title('滤波前的频谱');xlabel("频率/ \pi rad");ylabel("幅值");
subplot(4,1,2);plot(Wx/pi,abs(X_FIR1)); title('矩形窗低通FIR滤波后的频谱');xlabel("频率/ \pi rad");ylabel("幅值");
subplot(4,1,3);plot(Wx/pi,abs(X_FIR2)); title('哈明窗低通FIR滤波后的频谱');xlabel("频率/ \pi rad");ylabel("幅值");
subplot(4,1,4);plot(Wx/pi,abs(X_FIR3)); title('凯塞窗低通FIR滤波后的频谱');xlabel("频率/ \pi rad");ylabel("幅值");

figure(9);
subplot(2,1,1);plot(Wx/pi,abs(X_n)); title('滤波前的频谱');xlabel("频率/ \pi rad");ylabel("幅值");
subplot(2,1,2);plot(Wx/pi,abs(X_BP)); title('哈明窗带通FIR滤波后的频谱');xlabel("频率/ \pi rad");ylabel("幅值");

figure(10);
subplot(5,1,1);
plot(t,x);
title('原始语音信号的时域波形图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(5,1,2);
plot(t,x_n);
title('音频信号加带限噪声后的时域图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(5,1,3);
plot(t,x_IIR2);
title('IIR低通滤波器去噪后的时域图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(5,1,4);
plot(t,x_FIR2);
title('FIR低通滤波器去噪后的时域图');
xlabel("时间t/ s");
ylabel("幅值");
subplot(5,1,5);
plot(t,x_BP);
title('FIR带通滤波器去噪后的时域图');
xlabel("时间t/ s");
ylabel("幅值");


%sound(x,Fs);    %%听原始音频
%sound(x_n,Fs);  %%听加噪音频

%% 听去噪音频
% sound(x_IIR1,Fs); 
% sound(x_IIR2,Fs);  
% sound(x_IIR3,Fs); 
% sound(x_FIR1,Fs);  
% sound(x_FIR2,Fs); 
% sound(x_FIR3,Fs);
% sound(x_BP,Fs);

% clear sound  %%终止播放