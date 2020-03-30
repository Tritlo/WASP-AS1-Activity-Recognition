% 30/03/2020
% Assignment WASP HAR 
% Reading and parsing data

clc, clear; close all;

fID = fopen('sensorLog_20200326T184105e5xup4ssiuI_walking_withGPS.txt','r');
% fID = fopen('sensorLog_20200214T130304.txt','r');

A = textscan(fID,'%d %s %f %f %f %f');
% A = fscanf(fID,'%d %s %f %f %f',[Inf 5]); % does not work

T = length(A{3});

acc_x_data = [];
acc_y_data = [];
acc_z_data = [];

ori_x_data = [];
ori_y_data = [];
ori_z_data = [];

mag_x_data = [];
mag_y_data = [];
mag_z_data = [];

gps_lati_data = [];
gps_long_data = [];
gps_z_data = [];

for k = 1:T
    if strcmp(A{2}(k),'ACC')
        acc_x_data = [acc_x_data A{3}(k)];
        acc_y_data = [acc_y_data A{4}(k)];     
        acc_z_data = [acc_z_data A{5}(k)];           
    elseif strcmp(A{2}(k),'ORI')
        ori_x_data = [ori_x_data A{3}(k)];
        ori_y_data = [ori_y_data A{4}(k)];     
        ori_z_data = [ori_z_data A{5}(k)];           
    elseif strcmp(A{2}(k),'MAG')       
        mag_x_data = [mag_x_data A{3}(k)];
        mag_y_data = [mag_y_data A{4}(k)];     
        mag_z_data = [mag_z_data A{5}(k)];                   
    else
        gps_lati_data = [gps_lati_data A{3}(k)];
        gps_long_data = [gps_long_data A{4}(k)];     
        gps_z_data = [gps_z_data A{5}(k)];                   
    end
end

t_acc = 1:length(acc_x_data);
t_ori = 1:length(ori_x_data);
t_mag = 1:length(mag_x_data);
t_gps = 1:length(gps_lati_data);

figure(1);
plot(t_acc,acc_x_data,t_acc,acc_y_data,t_acc,acc_z_data);
title('Accelerometer');
xlabel('Time step');
ylabel('Amplitude');
legend('X Axis','Y Axis','Z Axis')
grid on;

figure(2);
plot(t_ori,ori_x_data,t_ori,ori_y_data,t_ori,ori_z_data);
title('Orientation');
xlabel('Time step');
ylabel('Amplitude');
legend('X Axis','Y Axis','Z Axis')
grid on;

figure(3);
plot(t_mag,mag_x_data,t_mag,mag_y_data,t_mag,mag_z_data);
title('Magnetometer');
xlabel('Time step');
ylabel('Amplitude');
legend('X Axis','Y Axis','Z Axis')
grid on;

figure(4);
subplot(2,1,1)
plot(t_gps,gps_lati_data);
title('GPS');
xlabel('Time step');
ylabel('Amplitude');
legend('Latitude')
grid on;
subplot(2,1,2)
plot(t_gps,gps_long_data);
xlabel('Time step');
ylabel('Amplitude');
legend('Longitude')
grid on;