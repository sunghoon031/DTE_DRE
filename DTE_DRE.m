clear all; close all; clc;

%% Notations:
% 1. xyz is the camera position viewed from the external reference frame.
% 2. R is the 3 x 3 matrix and its each column corresponds to the unit 
%    vector in x, y, z direction, viewed from the external reference frame.
%    This means that if the camera was rotated by some rotation M, its R
%    would be updated as M*R.


%% Generate a synthetic dataset
n_total = 100;
n_outliers = 2;
sigma_xyz = 0.05;
sigma_R = 5; %deg

[xyz_gt, R_gt, xyz_input, R_input] = GenerateSyntheticData(n_total, n_outliers, sigma_xyz, sigma_R);

              
%% Compute DTE
k = 5; % default = 5
[DTE, DRE] = ComputeDTE_DRE (k, xyz_gt, R_gt, xyz_input, R_input)






%%
function [xyz_gt, R_gt, xyz_input, R_input] = GenerateSyntheticData(n_total, n_outliers, sigma_xyz, sigma_R)

    % Ground truth points are inside 1x1x1 cube centered at the origin.
    xyz_gt = nan(3, n_total);
    R_gt = nan(3,3, n_total);

    for i = 1:n_total
        xyz_gt(:,i) = rand(3,1)-0.5;
        R_gt(:,:,i) = RandomRotation(rand(1)*360);
    end


    % Outlier points are inside 10x10x10 cube centered at the origin.
    xyz_outliers = nan(3, n_total);
    R_outliers = nan(3,3, n_total);

    for i = 1:n_total
        xyz_outliers(:,i) = (rand(3,1)-0.5)*10;
        R_outliers(:,:,i) = RandomRotation(rand(1)*360);
    end

    xyz_input = [xyz_gt(:,1:n_total-n_outliers), xyz_outliers(:, 1:n_outliers)];
    xyz_input = xyz_input + normrnd(0, sigma_xyz, size(xyz_input));

    R_input = nan(size(R_gt)); 
    R_transform = RandomRotation(rand(1)*360); % GT world to EST world
    s_transform = rand(1)*10;
    t_transform = rand(3,1)*100;

    for i = 1:n_total
        R_input(:,:,i) = RandomRotation(abs(normrnd(0, sigma_R)))*R_gt(:,:,i);
        R_input(:,:,i) = R_transform*R_input(:,:,i);

        xyz_input(:,i) = s_transform*R_transform*xyz_input(:,i) + t_transform;
    end
    R_input(:,:,n_total-n_outliers+1:end) = R_outliers(:,:,1:n_outliers);



end

function [DTE, DRE] = ComputeDTE_DRE (k, xyz_gt, R_gt, xyz_input, R_input)

    n_total = size(xyz_gt, 2);
    
    % (1)  Shift the ground-truth and estimated trajectories such that
    % both of their geometric medians are located at the origin.
    [xyz_med_gt, ~, median_error_gt] = GeometricMedian(xyz_gt , 10);
    [xyz_med_input, ~, median_error_input] = GeometricMedian(xyz_input , 10);
    
    xyz_gt = xyz_gt - xyz_med_gt;
    xyz_est = xyz_input - xyz_med_input;

    % (2) Rotate the estimated trajectory such that it minimizes the
    % sum of geodesic distances between the corresponding camera
    % orientations. 

    [R_geo1, ~, mean_error, rms_error] = AlignRotationL1(R_gt, R_input);

    for i = 1:n_total
        xyz_est(:,i) = R_geo1*xyz_est(:,i);
    end

    % (3) Scale the estimated trajectory such that the
    % median distance of the cameras to their geometric median is the
    % same as that of the ground truth.

    s_est = median_error_gt/median_error_input;

    for i = 1:n_total
        xyz_est(:,i) = s_est*xyz_est(:,i);
    end

    % (4)  Compute, winsorize and normalize the distances between the
    % corresponding cameras.

    errors = sqrt(sum((xyz_est - xyz_gt).^2 ,1));

    errors(errors>k*median_error_gt) = k*median_error_gt;

    errors = errors/(k*median_error_gt);

    % (5) Obtain the DTE by taking the average of the mean and
    % the root-mean-square (RMS) of the resulting distances.

    e_mean = mean(errors);
    e_rms = (sum(errors.^2)/n_total)^0.5;

    DTE = 0.5*e_mean + 0.5*e_rms;


    %% Compute DRE
    DRE = 0.5*mean_error+0.5*rms_error;



end

function [ x_med, mean_error, median_error ] = GeometricMedian(x , nIterations)

    % Input x is [3 x N]

    x_med = median(x,2);
    
    for it = 1:nIterations

        if (sum(sum(abs(x-x_med))==0) ~= 0)
            x_med = x_med+rand(size(x_med,1),1)*0.001;
        end

        step_num = 0;
        step_den = 0;

        for i = 1:size(x,2)
            v_norm = norm(x(:,i)-x_med);

            step_num = step_num + x(:,i)/v_norm;
            step_den = step_den + 1/v_norm;
        end


        x_med = step_num/step_den;
    end

    mean_error = mean(sqrt(sum((x-x_med).^2,1)));
    median_error = median(sqrt(sum((x-x_med).^2,1)));
    
end


function out = RandomRotation(angle_deg)
    axis = rand(3,1)-0.5;
    axis = axis/norm(axis);
    angle = angle_deg/180*pi;
    rotvec = angle*axis;
    out = ExpMap(rotvec);
end

function out = ExpMap(in)
    angle = norm(in);
    if (angle == 0)
        out = eye(3);
        return;
    end
    axis = in/angle;
    ss = SkewSymmetricMatrix(axis);
    R = eye(3)+ss*sin(angle)+ss^2*(1-cos(angle));
    out = R;
end

function out = LogMap(in)
    if (in(1,1) == 1 && in(2,2) == 1 && in(3,3) == 1)
        out = [0;0;0];
        return;
    end
    
    cos_theta = (trace(in)-1)/2;
    sin_theta = sqrt(1-cos_theta^2);

    if (sin_theta == 0)
        out = [0;0;0];
        return;
    end
    
    theta = acos(cos_theta);
    ln_R = theta/(2*sin_theta)*(in-in');
    out = [ln_R(3,2);ln_R(1,3);ln_R(2,1)];
end

function out = SkewSymmetricMatrix(in)
    out=[0 -in(3) in(2) ; in(3) 0 -in(1) ; -in(2) in(1) 0 ];
end

function [R_geo1, errors, mean_error, rms_error] = AlignRotationL1(R_true, R_est)

    % R_true and R_est are [3x3xN] matrices.
    
    nViews = size(R_true,3);
    
    errors = zeros(1,nViews);
    R_transform = cell(1, nViews);
    
    for i = 1:nViews
        R_transform{i} = R_true(:,:,i)*R_est(:,:,i)';
    end
  
    vectors_total = zeros(9,nViews);
    for i = 1:nViews
        vectors_total(:,i)= R_transform{i}(:);
    end
    med_vectors_total = median(vectors_total,2);
    [U,~,V] = svd(reshape(med_vectors_total, [3 3]));
    R_med = U*V.';
    if (det(R_med) < 0)
        V(:,3) = -V(:,3);
        R_med = U*V.';
    end

    
    R_geo1 = R_med;
    for j = 1:10
        step_num = 0;
        step_den = 0;
        for i = 1:nViews
            v =  LogMap(R_transform{i}*R_geo1');
            while (norm(v) < 1e-6)
                v = v + rand(3,1)*0.0001;
            end
            v_norm = norm(v);
            step_num = step_num + v/v_norm;
            step_den = step_den + 1/v_norm;
        end
        delta = step_num/step_den;
        delta_angle = norm(delta);

        
        
        delta_axis = delta/delta_angle;
        ss_delta = SkewSymmetricMatrix(delta_axis);
        R_delta = eye(3)+ss_delta*sin(delta_angle)+ss_delta^2*(1-cos(delta_angle));
        R_geo1 = R_delta*R_geo1;
    end
    
    

    for i = 1:nViews
        error = abs(acosd((trace(R_true(:,:,i)*(R_geo1*R_est(:,:,i))')-1)/2));
        errors(i) = error;
    end
    mean_error = mean(errors);
    rms_error = sqrt(mean(errors.^2));
end