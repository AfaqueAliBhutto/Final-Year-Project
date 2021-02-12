%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Delay and Sum Beamformer for Compound Plane Wave Imaging
% by Sevan Harput & Zainab Alomari
% University of Leeds, UK. November 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       
%       Performes DAS beamforming for the RF data of a plane wave steered
%       with (theta_d) degree.
%       RF_data is the received RF data.
%       theta_d is the steering angle of the plane wave in degree, can be
%       a positive or a negative value.
%       z_start is the imaging depth starting point in meter.
%       z_stop is the imaging depth ending point in meter.
%       image_width is the required width of the produced image.
%       lateral_step is the lateral step size between each lines of the
%       beamformed image. (pitch/2 is a good choice)
%       N_elements is the total number of elements in the transducer. 
%       pitch is the distance between the centres of two adjacent elements.
%       c is the sound speed im m/s.
%       fs is the sampling frequency.
%
%
%   Method:
%       In DAS beamforming, for each point of (x,z), the RF data from each receiving element is
%       delayed and the selected samples from these elements are summed. 
%       for each point (x,z), the delays applied to the RF data are
%       calculated by the time required for the signal to travel from the
%       transmiter to the field point and back to the receiving element as follows:
%       t = t_transmit + t_receive
%       t_transmit= [ z.cos(theta_d)+x.sin(theta_d)+0.5.N_elements.pitch.sin(|theta_d|) ]/c
%       t_receive(for element j)= sqrt(z^2 + (xj-x)^2)/c
%
%
%   Additions:
%       Receive Apodization - Hann window (twice the transducer) 
%       Element Directivity Check - discard contributions from elements that are >45 degrees to the beamforming point
%                                 - corrected for non-zero starting depth
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Example:
%       RF_data = UARP_TEMP_DATA;
%       theta_d = 0;
%       N_elements = 128;
%       pitch = 0.3048e-3;
%       image_width = N_elements*pitch;
%       lateral_step = pitch/2;
%       z_start = 5e-3;
%       z_stop = 60e-3;
%       c = 1482;
%       fs = 80e6;
%       [Beamformed_DATA, z_axis, x_axis] = CPWI_Beamformer(RF_data, theta_d, z_start, z_stop, image_width, lateral_step, N_elements, pitch, c, fs);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Beamformed_DATA, z_axis, x_axis] = CPWI_Beamformer_SEVAN(RF_data, theta_d, z_start, z_stop, image_width, lateral_step, N_elements, pitch, c, fs)
tic

if image_width > 2*N_elements*pitch
    error('Image width cannot be larger than 2 times the physical transducer size!')
end

lambda_fs = c/fs;
x_axis = -image_width/2 : lateral_step : image_width/2;     % Define image x-axis (lateral)
z_axis = z_start: lambda_fs/2 : z_stop;                     % Define image z-axis (axial)
Beamformed_DATA = zeros(length(z_axis), length(x_axis));    % Allocate memeory for the Beamformed Data

max_dimension = round(sqrt(image_width^2 + z_stop^2)/(lambda_fs/2)); 	% Calculate the diagonal length as the maximum image dimension (in samples)
RF_data_padded = [RF_data;  zeros(max_dimension-size(RF_data,1), size(RF_data,2))];     % Zero padding according to the maximum image dimension


% Apodization and Directivity Mask Matrices
RX_apo_axis = -3*N_elements*pitch/2 : lateral_step : 3*N_elements*pitch/2;      % Receive apodization matrix 3 times larger than the physical transducer size
RX_apo_size = length(RX_apo_axis);
RX_apodization = [zeros(1, round(RX_apo_size/6)) hann(round(4*RX_apo_size/6))' zeros(1, RX_apo_size-round(RX_apo_size/6)-round(4*RX_apo_size/6))];  % Hann window - twice the transducer size
% RX_directivity_mask = ones(length(z_axis), RX_apo_size);    
RX_directivity_mask = ones(round(z_stop/(lambda_fs/2)), RX_apo_size);    
mask_counter = 1;
for RX_apo_loc = RX_apo_axis	% Receive directivity mask 
    RX_directivity_mask(1:abs(round(RX_apo_loc/(lambda_fs/2))), mask_counter) = zeros(abs(round(RX_apo_loc/(lambda_fs/2))), 1);
    mask_counter = mask_counter + 1;
end    
% RX_directivity_mask(length(z_axis)+1:end, :) = [];	% Delete the excess elements
RX_directivity_mask(round(z_axis(end)/(lambda_fs/2))+1:end, :) = [];	% Delete the excess elements 
RX_directivity_mask(1: round(z_axis(1)/(lambda_fs/2))-1, :) = [];	% Delete the excess elements 

% Delay calculations and Beamforming
[X, Z] = meshgrid(x_axis, z_axis);
d1 = Z*cosd(theta_d) + X*sind(theta_d) + (N_elements * pitch/2)*sind(abs(theta_d)); % Distance from the transmitter to the points

for xj = 1 : 128  % Calculate the image data for each receiving element
    RF_address = round( ( d1 + sqrt(Z.^2+(X-(xj-(N_elements+1)/2)*pitch).^2) ) ./ lambda_fs );
    RF_col = RF_data_padded(:, xj);
    [RF_address_size1, RF_address_size2] = size(RF_address);
    Beamformed_DATA(1:RF_address_size1, 1:RF_address_size2) = Beamformed_DATA(1:RF_address_size1, 1:RF_address_size2) ...
        + RF_col(RF_address).*RX_directivity_mask(:,(1:RF_address_size2) + round((RX_apo_size-RF_address_size2)/2) - round((xj-(N_elements+1)/2)*pitch/lateral_step))...
        .*repmat(RX_apodization((1:RF_address_size2) + round((RX_apo_size-RF_address_size2)/2) - round((xj-(N_elements+1)/2)*pitch/lateral_step)), RF_address_size1, 1);    % repmat inside the loop to reduce the used memory (works faster)
end

x_axis = x_axis*1000;
z_axis = z_axis *1000;

toc

