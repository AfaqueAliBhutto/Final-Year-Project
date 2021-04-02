%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% New Performance Metrics for Ultrasound Pulse Compression Systems
% by Sevan Harput
% University of Leeds, UK. Copyright 2014.
% 
% Please use this code for scientific and educational purposes by refering
% to the publication below:
% Sevan Harput, James McLaughlan, David M. J. Cowell, and Steven Freear, 
% "New Performance Metrics for Ultrasound Pulse Compression Systems",
% IEEE International Ultrasonics Symposium, 2014.
% http://ieeexplore.ieee.org/xpl/articleDetails.jsp?tp=&arnumber=6931819
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function calculates the following performance evaluation metrics for
% a selected region of a log-compressed ultrasound image:
%   -3 dB mainlobe width (MLW3dB)
%   Full width at half maxiumum (FWHM) = -6 dB mainlobe width
%   Full width at half dynamic range (FWHDR) 
%   Peak sidelobe level (PSL)
%   Doubled mainlobe level (DML)
% FWHDR and DML are explained in the publication above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MLW3dB FWHM FWHDR PSL DML] = calculate_performance_metrics(region, resolution, dynamic_range)
% "region" can be either a 2D image or an axial slice of an image.
% "resolution" can be axial or lateral resolution.
%    Axial resolution = speed of sound / sampling frequency.
%    Lateral resolution = array element spacing (for linear imaging).
% "dynamic_range" is the image dynamic range in dB (e.g. 40).
% Units for MLW3dB, FWHM, and FWHDR are the same with the "resolution".
% Units for PSL and DML are in dB.

if isvector(region)
    signal = region(:)';    % region is in vector form
else
    signal = max(region');  % select an axial slice from the 2D image
end

[val peak_loc] = max(signal);	% location of the mainlobe
signal_norm = signal - val;     % normalized signal

% -3dB mainlobe width
MLwidth_3dB = find_XdB_width(signal_norm, peak_loc, 3);

% -6dB mainlobe width
MLwidth_FWHM = find_XdB_width(signal_norm, peak_loc, 6);

% Full width at half dynamic_range (FWHDR)
MLwidth_HDR = find_XdB_width(signal_norm, peak_loc, dynamic_range/2);

% Peak sidelobe level (PSL)
[pks,~] = findpeaks(signal_norm,'NPEAKS',2,'SORTSTR','descend'); 

% Doubled mainlobe level (DML)
DML_search = -dynamic_range/2;     % search starting point
search_step_size = 0.1;     % search step size = 0.1 dB (can be reduced for precision)
search_width = 0;           % initialize search parameter
flag = 0;                   % initialize search parameter
while ( sqrt(2) * MLwidth_HDR >= search_width )	% search loop for the DML
    DML_search = DML_search - search_step_size;
    search_width = find_XdB_width(signal_norm, peak_loc, -DML_search);
    if ( numel(search_width) == 0);
        flag = 1; break; end
end


% Performance metrics 
MLW3dB = MLwidth_3dB * resolution;
FWHM = MLwidth_FWHM * resolution;
FWHDR = MLwidth_HDR * resolution;

if numel(pks) <= 1
    PSL = 'No sidelobe exists.';
else
    PSL = pks(2);
end

if (flag == 1 || numel(FWHDR) == 0)
    DML = 'Mainlobe width is not doubled within the selected region.';
else
    DML = DML_search;
end

end



%% Search function for the X dB width
function [MLwidth] = find_XdB_width(signal_norm, peak_loc, XdB)

dB_point_pos = find(signal_norm(peak_loc:end) <= -XdB, 1, 'first') - 1;
dB_point_neg = find(fliplr(signal_norm(1:peak_loc)) <= -XdB, 1, 'first') - 1;

% Linear interpolation to find the exact -XdB point
dB_point_pos_interp = dB_point_pos + ...
    ((-XdB - signal_norm(peak_loc+dB_point_pos)) / (signal_norm(peak_loc+dB_point_pos) - signal_norm(peak_loc+dB_point_pos-1)));
dB_point_neg_interp = dB_point_neg + ...
    ((-XdB - signal_norm(peak_loc-dB_point_neg)) / (signal_norm(peak_loc-dB_point_neg) - signal_norm(peak_loc-dB_point_neg+1)));

if (numel(dB_point_pos_interp) == 0 || numel(dB_point_neg) == 0)
    MLwidth = [];
else
    MLwidth = dB_point_pos_interp + dB_point_neg_interp;
end
end
