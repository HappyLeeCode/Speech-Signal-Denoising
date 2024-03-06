function [y,hpf_noise,noise] = Gnoisegen(x,Fs,snr)
%% 本函数功能：产生一个高频的高斯白噪声，并根据给定的信噪比，将高频噪声与输入信号进行混频合成
%  信噪比不同，即噪声强度不同

%% 输入参数 x:原始语音信号        Fs:信号的采样频率       snr:信噪比
%% 输出参数 y:最终合成的带噪信号  hpf_noise:高频高斯噪声  noise:全通高斯噪声

Nx = length(x);      % 求出信号x长度

% 用randn函数产生高斯白噪声
noise = randn(Nx,1); 
                               
% 高通滤波器设计（切比雪夫2型）
Fstop = Fs/4;        % 阻带截止频率，即归一化频率为0.5π处
Fpass = Fs/4+175;    % 通带截止频率
Astop = 80;          % 阻带最小衰减（dB）
Apass = 1;           % 通带最大衰减（dB）
match = 'stopband';  
h  = fdesign.highpass(Fstop, Fpass, Astop, Apass, Fs);
Hd = design(h, 'cheby2', 'MatchExactly', match);

% 产生高频高斯噪声
hpf_noise = filter(Hd,noise);

% 根据信噪比snr来合成带噪语音
signal_power = 1/Nx*sum(x.*x);                              % 求出信号的平均能量
noise_power = 1/Nx*sum(hpf_noise.*hpf_noise);               % 求出噪声的能量
noise_variance = signal_power / ( 10^(snr/10) );            % 计算出噪声设定的方差值
hpf_noise = sqrt(noise_variance/noise_power)*hpf_noise;     % 按噪声的平均能量构成相应的白噪声
y = x+hpf_noise;                                            % 合成带噪语音

end
