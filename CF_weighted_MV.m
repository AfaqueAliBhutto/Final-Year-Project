%   Inputs:
%       RF_data is the received RF data.
%       theta_d is the steering angle of the plane wave in degree, can be
%       a positive or a negative value.
%       z_start is the imaging depth starting point (in meters).
%       z_stop is the imaging depth ending point (in meters).
%       image_width is the required width of the produced image.
%       W is the total width of the transducer (in meters), where: W=N_elements x pitch.
%       lambda is the wavelength.
%       N_elements is the total number of elements in the transducer that
%       are used to transmit the ultrasound beam.
%       pitch is the distance between the centres of two adjacent elements
%       (in meter).
%       c is the sound speed im m/sec.
%       Ts is the sampling time.
%       td is the length of segment taken from each element during the
%       beamforming. The centre of this segment resembles the response
%       received from the focal point. To retain axial resolution, td
%       should not exceed the pulse length (which is the convolution
%       between the excitation signal and the 2-way impulse response).
%       (td is always an odd number).
%       Lp is the number of elements in each subarray. Increasing Lp gives
%       higher resolution on the cost of lowering robustness. Decreasing Lp
%       reduces the resolution and improves the robustness, where Lp=1
%       resembles DAS beamformer with uniform aperture shading (unity apodization).
%       dx is the resolution (or step) in x direction.
%
%   Returns:
%           Beamformed_data: The frame resulted from MV beamforming operation.
%
%   Notes:
%      In MV beamforming, a vector of length (td) is selected from each
%      element's RF-data, where the response from the focal point is at the
%      centre point of this vector.
%      For each point (x,z), the delays applied to the RF data are
%      calculated by the time required for the signal to travel from the
%      transmitter to the field point and back to the receiving element as follows:
%      t= t_transmit + t_receive
%      t_transmit= [ z.cos(theta_d)+x.sin(theta_d)+0.5.W.sin(|theta_d|) ]/c
%      t_receive(for element j)= sqrt(z^2 + (x-xj)^2)/c
%      (where xj is the axial distance of the j-th element)

%      After performing the delays, adpative weight is calculated for each
%      point according to the equation that ensures minimizing the output
%      power while preserving a unity gain in the focal point. This equation is: 
%      w=(inv(R).e)/(conj(e)'.inv(R).e)
%      where e is a vector of ones. R is the covariance matrix.
%      After calculating the weight, the output value is calculated using:
%      B=conj(w)'.summation(Gp)/P
%      where P is the number of subarrays and Gp is the subarray.

%       Diagonal Loading can be used to increase the robutness of the
%       beamformer by adding a constant value to the diagonal of the
%       covariance matrix using the following equation:
%       R=R+ delta.trace{R}.I
%       where delta is 1/(DL.Lp), DL is the diagonal loading coefficient,
%       I is the identity matrix.

%      The resulted image dimensions are specified by the user by z_start,
%      z_stop and image_width. For compounding 3 angles for example,
%      this function will be called 3 times and the three reults are
%      averaged, normalized to the maximum value, envelope detected
%      by hilbert transform and then converted to dB.

% References:
% 1) J-F Synnevag et al., "Adaptive beamforming applied to medical
 ultrasound imaging", Ultrasonics, Ferroelectrics and Frequency Control,
 IEEE Transactions on, 54(8):1606-1613, 2007.
% 2) I. K. Holfort et al., "Broadband minimum variance beamforming for
% ultrasound imaging", Ultrasonics, Ferroelectrics and Frequency Control, 
% IEEE Transactions on, 56(2):314-325, 2009.


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


%% Transducer definition  96-element linear array transducer
% 
% Our next step is to define the ultrasound transducer array we are using.
% For this experiment, we shall use the 96 element
% Transducer and set our parameters to match it.
    
f0                      =   4e6;              % Transducer center frequency [Hz]
c                       =   1540;             % Speed of sound [m/s]
fs                      =   100e6;            % Sampling frequency [Hz]
Ts                      =   1/fs;             % Sampling step [s]
lambda                  =   c/f0;             % Wavelength [m]
height                  =   6e-3;             % Height of element [m]
L                       =   c*Ts/2;           % L is the distance between each 2 points in the z direction
pitch                   =   0.3048e-3;        % probe.pitch [m]
kerf                    =   pitch/10;         % gap between elements [m]
width                   =   pitch-kerf;       % Width of element [m]
N_elements              =   96;               % Number of elements



%% %%%%%%%%%%%%Initial electronic focus%%%%%%%%%%%%%%%

    focus=[0 0 100]; % Initial electronic focus
    
    Th = xdc_linear_array (N_elements, width, height, kerf, 1,1, focus);
    Th2 = xdc_linear_array (N_elements, width, height, kerf, 1,1, focus);
    
    BW=1*f0;
    
    impulse_response=sin(2*pi*f0*(0:Ts:2/BW));
    impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
    

    xdc_impulse (Th, impulse_response);
    xdc_impulse (Th2, impulse_response);
    
    excitation=sin(2*pi*f0*(0:Ts:2/BW));
    excitation=excitation.*hanning(max(size(excitation)))';
    
  
    xdc_excitation (Th, excitation);
    
    LengthConv=(length(excitation) + 2*length(impulse_response)-2)*0.5;

%%
dx            = lambda/10;
imaging_start = 29/1000;
imaging_depth = 2/1000; % 30/1000; 
image_width   = 5e-3;
L             = c*Ts/2; % L is the distance between each 2 points in the z direction
z_start       = imaging_start;
z_stop        = imaging_start+imaging_depth;
X             = -image_width/2: dx: image_width/2;
Z             = z_start: L : z_stop;  


%% %%%%%%%%%%%% Scattering Points %%%%%%%%%%%%%%%

positions = [0 0 30]/1000; 
amp       = 1; 

td        = LengthConv*2; % this is one pulse length

tdd       = (td-1)/2;

Lp    = N_elements/2;       %Lp is the number of elements in each subarray.
P     = N_elements-Lp+1;    % the number of subarrays
e     = ones(Lp,1);         % the steering vector. 

yy    = zeros(25000,10000); % initial matrix for addition

W     = (N_elements*pitch); % -kerf

Tapo  = hamming(N_elements)'; %Transmit Apodization

xdc_apodization (Th, 0, Tapo);

        
 %%      
 %%%%%%%%% SETTING THE DELAYS %%%%%%%%%%%
    n=1:N_elements;
    N_angles=1;
    for theta_d=0:5:0
        delays=n*pitch*sind(theta_d)/c;
        delays=delays-min(delays);
        xdc_focus_times(Th, 0, delays);

        [v1,times]=calc_scat_multi(Th, Th2, positions, amp);
        
        v1_zeropadded= [zeros(round(times*fs) , N_elements); v1];
        
        v(1:length(v1_zeropadded),:)=v1_zeropadded;
        
        %Pass the signal through an AWGN channel having a 60 dB signal-to-noise ratio (SNR)
        %adding white Gaussian noise
         RF_data=awgn( v, 60, 'measured');
         
         
%%         
        %%%%%%%%%%% Beamforming %%%%%%%%%%%%
        time_array=1:size(RF_data,1);
        Z=z_start: L :z_stop;
        X=-image_width/2: dx: image_width/2;
        tdd_matrix=repmat((-tdd:tdd) ,length(Z),1); % size of this matrix is (length(Z) x td)
        %%%%% Memory allocation:
        Ym=zeros(length(Z), td, N_elements);
        Beamformed_data=zeros(length(Z), length(X));
%         cf_matrix=zeros(length(Z), length(X));
        adrs_x=1;
        for x=-image_width/2:dx:image_width/2
            d1= Z.*cosd(theta_d)+x*sind(theta_d)+(W/2)*sind(abs(theta_d));% the distance from the transmitter to the points that have the same lateral distance of x.
%             d1=Z; % d1 becomes only Z when transmitting with 0 angle. 
            xjj=1;
            for xj=-(N_elements/2)+0.5:(N_elements/2)-0.5
                address_vector=( (d1+sqrt(Z.^2+(x-(xj*pitch))^2))/c/Ts ) +LengthConv;
                address_matrix= repmat(address_vector',1,td);  % size of this matrix is (length(Z) x td)
                final_matrix=address_matrix + tdd_matrix;% a matrix with a td-length address vector for each point of x lateral distance.
                Ym(:,:,xjj)=interp1(time_array,RF_data(:,xjj),final_matrix,'linear',0);%Ym contains a td-length vector for each point of x lateral distance for each element
                xjj=xjj+1;
            end
 %%
 %%%%%%%%% now calculating the weight (w) for each point:
            for adrs_z=1:length(Z)
                R=zeros(Lp,Lp);
                sum_Gp=zeros(Lp,td);
              

                for loopG = 1 : P
                    Gp=(shiftdim(Ym(adrs_z , : , loopG:loopG+Lp-1 ),1))'; % Converting from 3d to 2d.
                    R=R+ Gp*conj(Gp');
                    sum_Gp=sum_Gp+Gp; 
                end
                
                R=R/P;
                
                %trace(A) is the sum of the diagonal elements of A, which is also the sum of the eigenvalues of A.
                %I = eye(n) returns an n-by-n identity matrix with ones on the main diagonal and zeros elsewhere.
                
                R=R+(1/(20*Lp))*trace(R)*eye(Lp); %applying Diagonal Loading to R before calculating the weight.
                
                w=(R \ e) / (conj(e') * (R \ e));%  .* hann(Lp) ; % calculating the weight, w=(Lp x 1)
                

   
%%                 %%%%%%% CF %%%%%%%%%%%%%
                S=Ym(adrs_z, tdd+1 , :);
                CF=(abs(sum(S)))^2  / (N_elements* sum(abs(S))^2);
                B= CF*conj(w)' * (sum_Gp / P); % conj(CF) * (sum_Gp / P);  %   %conj(GCF) * (sum_Gp / P);   %CF * conj(w)' * (sum_Gp / P); 
                Beamformed_data(adrs_z,adrs_x)=B((td+1)/2); % taking the central point of the output vector.
            end
            adrs_x=adrs_x+1;
        end
    yy(1:size(Beamformed_data,1),1:size(Beamformed_data,2))=yy(1:size(Beamformed_data,1),1:size(Beamformed_data,2))+Beamformed_data;
    end
    
%%

    yy1=yy(1:size(Beamformed_data,1),1:size(Beamformed_data,2))/N_angles;  % averaging
    
    %%************************
    env_dB=20*log10(abs(hilbert(yy1)));

    env_dB=env_dB-max(env_dB(:));

%%
%%%%%%%%%%%%%%%%% Display %%%%%%%%%%%%%%%%%%%%%%
    depth=z_start:L:z_stop;
    x=-image_width/2:dx:image_width/2;
 

%%    
 %%%%%%%%%%%%%%%%%%% To calculate Lateral Resolution:%%%%%%%%%%%%%%%%
        dx = kerf*1e6;
        
[MLW3dB, FWHM , FWHDR , PSL , DML] = calculate_performance_metrics(env_dB(:,round(163/2)), dx, 60); 
   
Lat_r = FWHM;
X = sprintf('Leteral Resolution is: %f',Lat_r);
disp(X)
      


     
 %% 
  %%%%%%%%%%%%%%%%%%% To calculate Axial Resolution:%%%%%%%%%%%%%%%%
        dx = (c/fs)*1e6;
        
[MLW3dB, FWHM , FWHDR , PSL , DML] = calculate_performance_metrics(env_dB(round(260/2),:), dx, 60); 
   ax_r = FWHM;
Y = sprintf('Axial Resolution is: %f',ax_r);
disp(Y)

c = categorical({'Lateral Resolution','Axial Resolution'});
a = [Lat_r,ax_r];
figure;bar(c,a)

%%%%%%%%%%%%%%%%%%%%
    figure;
    imagesc(x*1000, depth*1000, env_dB, [-50 0])
    xlabel('Lateral distance [mm]')
    ylabel('Depth [mm]')
    title(['Lateral Resolution is: ' num2str(Lat_r) ' and Axial Resolution is: ' num2str(ax_r)]);
   % title(['CFMV: N=' num2str(N_elements) ' f0=' num2str(f0*1e-6) 'MHz,td=' num2str(td) ',Lp=' num2str(Lp)]);
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


  
  




   
  
