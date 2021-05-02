


%% Clear old workspace and close old plots
clc
clear all;
close all;

tic


%% field II initialisation
% 
% Next, we initialize the field II toolbox. Again, this only works if the 
% Field II simulation program (<field-ii.dk>) is in MATLAB's path. We also
% pass our set constants to it.

path (path,'E:\Ultrasound_Beamforming\Field_II_combined' )

field_init;


%% Transducer definition  128-element linear array transducer
% 
% Our next step is to define the ultrasound transducer array we are using.
% For this experiment, we shall use the 128 element
% Transducer and set our parameters to match it.
    
f0                      =   4e6;              % Transducer center frequency [Hz]
c                       =   1540;             % Speed of sound [m/s]
fs                      =   100e6;            % Sampling frequency [Hz]
Ts                      =   1/fs;             % Sampling step [s]
lambda                  =   c/f0;             % Wavelength [m]
lambda_fs               =   c/fs;
height                  =   6e-3;             % Height of element [m]
L                       =   c*Ts/2;           % L is the distance between each 2 points in the z direction
pitch                   =   0.3048e-3;        % probe.pitch [m]
kerf                    =   pitch/10;         % gap between elements [m]
width                   =   pitch-kerf;       % Width of element [m]
N_elements              =   128;               % Number of elements


 

%% 
%%%%%%%%%%%%Initial electronic focus%%%%%%%%%%%%%%%

focus =[0 0 100]; 

Th  = xdc_linear_array (N_elements, width, height, kerf, 1,1, focus);
Th2 = xdc_linear_array (N_elements, width, height, kerf, 1,1, focus);

BW=2*f0;

impulse_response=sin(2*pi*f0*(0:1/fs:2/BW));
impulse_response=impulse_response.*hanning(max(size(impulse_response)))';


xdc_impulse (Th, impulse_response);
xdc_impulse (Th2, impulse_response);

BW=2e6/f0;

% the BW here specifies only the duration of the total pulse but not the amount of energy inside the envelope, 
% i.e. it pads zeros to reach this duration but no change on the signal.
tc=gauspuls('cutoff' ,f0 ,BW,[],-60); 
tt=-tc:Ts:tc;
excitation = gauspuls(tt, f0, BW); % the BW here specifies the duration of the pulse or the number of sycles in the pulse.

ax1 = subplot(2,1,1); % top subplot
plot(ax1,tt, excitation);
title(ax1,['Exitation Pulse']);


xdc_excitation (Th, excitation);
Half_signal=(length(excitation) + 2*length(impulse_response)-2)*0.5;




   

%% 
%%%%%%%%%%%%%%%%%% Scattering Points phantom created manually %%%%%%%%%%%%%%%%%%
% [positions, amp] = cyst_ph(100000);

positions       = [0 0 30]/1000;
amp             = 1;
imaging_start   = 29/1000; % (A-3)/1000;
imaging_depth   = 2/1000; % 6/1000;
image_width     = 6e-3;
dx              = lambda/20;

yy              = zeros(25000,10000); % initial matrix for addition
W               = (N_elements*pitch); % -kerf
L               = c*Ts/2; % L is the distance between each 2 points in the z direction
z_start         = imaging_start;
z_stop          = imaging_start+imaging_depth;
N_activeT       = N_elements;
Tapo            = hamming(N_activeT)';

ax2 = subplot(2,1,2); % bottom subplot
plot(ax2,Tapo);
title(ax2,['Apodization']);

xdc_apodization(Th, 0, Tapo); %Procedure for creating an apodization time line for an aperture

n=1:N_elements;


 %% 
 %%%%%%%%%%%%%%%% SETTING THE DELAYS %%%%%%%%%%%%%%%%
   
    N_angles=1;
    
for thetad=0:3:0 
        delays=n*pitch*sind(thetad)/c;
        delays=delays-min(delays);
        xdc_focus_times(Th, 0, delays);

        %%%%%%%%% using calc_scat %%%%%%%%%%%%%%%%
        
        v=zeros(8000,N_activeT);
        
        for i=1:N_activeT
            Rapo=[zeros(1, i-1) 1 zeros(1, N_elements-i)];
            xdc_apodization (Th2, 0, Rapo);
            [v1,t1]=calc_scat(Th, Th2, positions, amp);
            v1_zeropadded = [zeros(1,round(t1*fs))'; v1];
            v(1:length(v1_zeropadded),i) = v1_zeropadded;
        end
  

time_array = 1:size(v,1);
x          = -image_width/2 : dx : image_width/2 ;
z          = z_start: L : z_stop ;
ff         = zeros(length(z), length(x));
[X, Z]     = meshgrid(x,z);
d1         = Z; %*cosd(theta_d)+X*sind(theta_d)+(W/2)*sind(abs(theta_d)); % the distance from the transmitter to the points    

elements_counter=1;

for xj= -(N_elements/2)+0.5 : (N_elements/2)-0.5  % loop for the receiving elements
    RF_address= (( d1+sqrt(Z.^2+(X-xj*pitch).^2) ) ./ (c*Ts))+Half_signal;
    ff= ff+ interp1(time_array,v(:,elements_counter),RF_address, 'linear', 0);
    elements_counter=elements_counter+1;
end

%%%%%%%%%%%%%% Hilbert Transform %%%%%%%%%%%%%%
        yy(1:size(ff,1),1:size(ff,2))=yy(1:size(ff,1),1:size(ff,2))+ff;
  
end
    
    ab1=yy(1:size(ff,1),1:size(ff,2))/N_angles;
    env=abs(hilbert(ab1/max(max(ab1))));
    
 
   
   %% 
 %%%%%%%%%%%%%%%%% Display %%%%%%%%%%%%%%%%%%%%%%
    
    env_dB  =  20*log10(env);
    depth   =  z_start:L:z_stop;
    x       =  -image_width/2:dx:image_width/2;
   
   



%profil_ligne = 20*log10(abs(hilbert(final_t1)));
%figure;plot(profil_ligne(2623,:))
%figure;plot(profil_ligne(3662,:))
%%    
 %%%%%%%%%%%%%%%%%%% To calculate Lateral Resolution:%%%%%%%%%%%%%%%%
        dx = kerf*1e6;
        
[MLW3dB, FWHM , FWHDR , PSL , DML] = calculate_performance_metrics(env_dB(:,round(390/2)), dx, 60); 
   
Lat_r = FWHM;
X = sprintf('Leteral Resolution is: %f',Lat_r);
disp(X)

     
 %% 
  %%%%%%%%%%%%%%%%%%% To calculate Axial Resolution:%%%%%%%%%%%%%%%%
        dx = (c/fs)*1e6;
        
[MLW3dB, FWHM , FWHDR , PSL , DML] = calculate_performance_metrics(env_dB(round(260/2),:), dx, 60); 
   ax_r = FWHM;
X = sprintf('Axial Resolution is: %f',ax_r);
disp(X)

c = categorical({'Lateral Resolution','Axial Resolution'});
a = [Lat_r,ax_r];
figure;bar(c,a)

%%
 figure;
    
    imagesc(x*1000, depth*1000, env_dB, [-50 0])
    
    xlabel('Lateral distance [mm]')
    ylabel('Depth [mm]')
    title(['Lateral Resolution is:' num2str(Lat_r) '  and   Axial Resolution is:' num2str(ax_r)]);
%     title('Four scattering points at 5, 10, 15 and 20cm depths'); %(['Aperture Width= ' num2str(apwidth*1e2) 'cm'])
    axis('image')
    colormap(gray(128));
    colorbar;
    
    toc
    
    
    %% Measure contrast
%
% Lets measure the contrast using the "contrast ratio" as our metric.

% First we need to put our images in a different data struct that the 
% measure contrast function expects
%images = env_dB;


% Define the coordinates of the regions used to measure contrast
%xc_nonecho = 0;      % Center of cyst in X
%zc_nonecho = 30;      % Center of cyst in Z
%r_nonecho = 0.3;        % Radi of the circle in the cyst
%r_speckle_inner = 0.5;  % Radi of the inner circle defining speckle region
%r_speckle_outer = 0.6;    % Radi of the outer circle defining speckle region

%axi = gca;
%viscircles(axi,[xc_nonecho,zc_nonecho],r_nonecho,'EdgeColor','r');
%viscircles(axi,[xc_nonecho,zc_nonecho],r_speckle_inner,'EdgeColor','y');
%viscircles(axi,[xc_nonecho,zc_nonecho],r_speckle_outer,'EdgeColor','y');

% Call the "tool" to measure the contrast
%[CR] = measure_contrast_ratio(env_dB,images,xc_nonecho, zc_nonecho,r_nonecho,r_speckle_inner,r_speckle_outer);

% Plot the contrast as a bar graph together with the two images
%figure(3);hold on
%subplot(2,3,[3 6]);
%bar(CR)   
%title('Measured Contrast');
%ylabel('CR');


% Manual ROI selection







    








