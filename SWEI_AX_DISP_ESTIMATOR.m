

function result = SWEI_AX_DISP_ESTIMATOR(fileName, ExpName, Exp)


load(fileName);
[~, ~, FramesN] = size(UARP_TEMP_DATA);
SSI_MID   = UARP_TEMP_DATA(:,:,[1:10, 17:end]);

DATA_NAME = ExpName;


%%      BEAMFORMING - FILTERING

% Beamforming
BF.c                        = 1540;
BF.fs                       = 80E6;
BF.f0                       = 4.79E6;
BF.spatial_sample           = (BF.c*(1/BF.fs) /2 );
BF.N_elements               = 128;
BF.pitch                    = 0.3048E-3;
BF.image_width              = BF.N_elements*BF.pitch + 10E-3;
BF.lateral_step             = BF.pitch/2;
BF.z_start                  = 1E-3;
BF.z_stop                   = Exp.ImagingDepth;
time_zero_samples           = Exp.TimeZero;

% Frequency Filter
order_1 = 30;
l_cutoff = 3E6;
u_cutoff = 8E6;
wl = l_cutoff / (BF.fs/2);
wu = u_cutoff / (BF.fs/2);
Filter_coefficients = fir1(order_1, [wl wu]);



% Spatial Filter
order_2 = 30;
lambda = BF.c/BF.f0;
Spatial_fstop = lambda/2;
w_s = 1 / (Spatial_fstop / (BF.lateral_step/2));
Filter_Coeffs_2 = fir1(order_2, w_s);


close all;
clc;
clear BF_DATA_MID BF_DATA_MID_filt1 BF_DATA_MID_filtered2;


Frames = [1:10, 17:FramesN]*0.1; % ms
Angles = Exp.angles;
Angle_Index = 1;

for frame = 1:size(SSI_MID, 3)
    
    FRAME = SSI_MID(:,time_zero_samples:end,frame)';
    [BF_DATA_MID, z_axis, x_axis] = CPWI_Beamformer_SEVAN...
        (FRAME,...
        Angles(Angle_Index),...
        BF.z_start,...
        BF.z_stop,...
        BF.image_width,...
        BF.lateral_step,...
        BF.N_elements,...
        BF.pitch,...
        BF.c,...
        BF.fs);
    
    
    for lat = 1:size(BF_DATA_MID,2)
        filtered = conv( BF_DATA_MID(:, lat), Filter_coefficients);
        BF_DATA_MID_filt1(:, lat) = filtered(1+order_1/2:end-order_1/2);
    end
    
    for axial = 1:size(BF_DATA_MID,1)
        filtered = conv( BF_DATA_MID_filt1(axial, :), Filter_Coeffs_2);
        BF_DATA_MID_filtered2(axial, :, frame) = filtered(1+order_2/2:end-order_2/2);
    end
    
    if frame == 1
        BF_Envelope_dB = 20*log10(abs(hilbert(BF_DATA_MID_filtered2(:,:,frame)/max(max(BF_DATA_MID_filtered2(:,:,frame))))));
        figure(frame);
        imagesc(x_axis, z_axis, BF_Envelope_dB, [-40 0]);
        colormap(gray); colorbar
        axis image
        xlabel('\bf X (mm)'                             , 'FontName', 'MonoSpaced');
        ylabel('\bf Z (mm)'                             , 'FontName', 'MonoSpaced');
        title(['B-MODE- ', DATA_NAME]                   , 'FontName', 'MonoSpaced');
        htxtbox = annotation('textbox',[0.57 0.7 0.1 0.1],'String',{[ num2str(Frames(frame)) ' ms']},...
            'FontSize',10, 'FontName','MonoPace','FitBoxToText','off','LineStyle','none','Color',[0 1 0]);
        htxtbox = annotation('textbox',[0.25 0.7 0.1 0.1],'String',{['Angle# ' num2str(Angles(Angle_Index))]},...
            'FontSize',10, 'FontName','MonoPace','FitBoxToText','off','LineStyle','none','Color',[0 1 0]);
        %         h = gcf;
        figname = (['B-Mode - ', DATA_NAME]);
        %         print(figname, '-dpng', '-r0');
        savefig(figname);
    end
    
    if Angle_Index == 3
        Angle_Index = 1;
    else
        Angle_Index = Angle_Index + 1;
    end
end


close all;
end