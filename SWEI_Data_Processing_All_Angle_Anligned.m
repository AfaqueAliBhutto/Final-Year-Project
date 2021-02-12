
clear;
close all;
clc;


cd('C:\Users\Afaque\OneDrive\Desktop\thesiss');
addpath(genpath('C:\Users\Afaque\OneDrive\Desktop\thesiss'));

Exp.ImagingDepth = 41E-3;
Exp.FrameRate    = 10E3;
Exp.TimeZero     = 210; 
Exp.angles       = [-2 0 +2];
fileName = 'DPB-HighElas-Zero-Aligned.mat';
ExpName  = 'HighElas-Zero-Aligned';
Result = SWEI_AX_DISP_ESTIMATOR(fileName, ExpName, Exp);

