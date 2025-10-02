clear; close all; clc;

% The following code finds contour, extracts end/turning points, and
% plots keypoints of a image set
%% MATCHING (train vs test image)
d = 10; 
L = 15;     % phi kernel size (odd)
weight = 1; % weight for rank + theta terms

images = cell(50, 1);
for n = 1:25
    images{n} = imread(sprintf("D:/專題/Character Verification/2/database/base_1_%d_2.bmp", n));
    images{n+25} = imread(sprintf("D:/專題/Character Verification/2/testcase/word_%d_1_2.bmp", n));
end

%% PROCESS & PLOT IN BATCHES OF 10
batchSize = 10;
numImgs = 50;

for batch = 1:ceil(numImgs/batchSize)
    figure;
    idxStart = (batch-1)*batchSize + 1;
    idxEnd   = min(batch*batchSize, numImgs);

    for n = idxStart:idxEnd
        rgbdouble = double(images{n});
        Y = 0.299*rgbdouble(:,:,1) + 0.587*rgbdouble(:,:,2) + 0.114*rgbdouble(:,:,3);
        B = Y < 220;
        C = findContours(B);

        end_xy_all = [];
        turn_xy_all = [];

        for ci = 1:numel(C)
            ctr = C{ci};
            if size(ctr,1) < 2*d, continue; end
            [end_idx, turn_idx] = classifyPoints(ctr, d);
            end_xy_all = [end_xy_all; ctr(end_idx,:)]; %#ok<AGROW>
            turn_xy_all = [turn_xy_all; ctr(turn_idx,:)]; %#ok<AGROW>
        end

        % --- Plotting ---
        subplot(2,5,n-idxStart+1); % 2×5 grid for 10 images
        imshow(images{n}); hold on; title(sprintf('Img %d', n));

        % Endpoints = green
        plot(end_xy_all(:,2), end_xy_all(:,1), 'go', 'MarkerSize', 4, 'LineWidth', 1.0);

        % Turning points = blue
        plot(turn_xy_all(:,2), turn_xy_all(:,1), 'bo', 'MarkerSize', 4, 'LineWidth', 1.0);
    end
end


%% FUNCTIONS
function contours = findContours(B)
    % Pad image to avoid boundary issues
    B = padarray(B, [1 1], 0);
    [rows, cols] = size(B);

    % Reference Directions (UL, U, UR, R, DR, D, DL, L)
    dirs = [ -1 -1; -1 0; -1 1; 0 1; 1 1; 1 0; 1 -1; 0 -1 ];
    visited = false(rows, cols);
    contours = {};

    for c = 2:cols-1
        for r = 2:rows-1
            if B(r,c) == 1 && B(r,c-1) == 0 && ~visited(r,c)
                contour = [];
                r_start = r; c_start = c;
                r_curr = r_start; c_curr = c_start;
                dir_idx = 4;
                contour = [contour; r_curr, c_curr];
                visited(r_curr,c_curr) = true;

                while true
                    switch dir_idx
                        case 1, order = [6 7 8 1 2 3 4 5]; % UL
                        case 2, order = [7 8 1 2 3 4 5 6]; % U
                        case 3, order = [8 1 2 3 4 5 6 7]; % UR
                        case 4, order = [1 2 3 4 5 6 7 8]; % R
                        case 5, order = [2 3 4 5 6 7 8 1]; % DR
                        case 6, order = [3 4 5 6 7 8 1 2]; % D
                        case 7, order = [4 5 6 7 8 1 2 3]; % DL
                        case 8, order = [5 6 7 8 1 2 3 4]; % L
                    end

                    found = false;
                    for k = 1:8
                        idx = order(k);
                        r_next = r_curr + dirs(idx,1);
                        c_next = c_curr + dirs(idx,2);
                        if B(r_next,c_next) == 1
                            contour = [contour; r_next, c_next];
                            visited(r_next,c_next) = true;
                            r_curr = r_next; c_curr = c_next;
                            dir_idx = idx;
                            found = true;
                            break;
                        end
                    end

                    if r_curr == r_start && c_curr == c_start, break; end
                    if ~found, break; end
                end
                contours{end+1} = contour - 1;
            end
        end
    end
end


function [endpoints, turningpoints] = classifyPoints(contour, d)
    N = size(contour,1);
    if N < 2*d
        endpoints = [];
        turningpoints = [];
        return;
    end
    angles = computeContourAngles(contour, d);
    endpoints = []; turningpoints = [];
    for k = 1:N
        theta_k = angles(k);
        is_min = true;
        for tau = -d:d
            if tau == 0, continue; end
            idx = mod(k + tau - 1, N) + 1;
            if ~(theta_k < angles(idx))
                is_min = false; break;
            end
        end
        if is_min
            if theta_k <= pi/6
                endpoints(end+1) = k;
            elseif theta_k < 5*pi/6
                turningpoints(end+1) = k;
            end
        end
    end
end


function angles = computeContourAngles(contour, d) % For classification
    N = size(contour,1);
    angles = zeros(N,1);
    for k = 1:N
        idx1 = mod(k-d-1, N) + 1;
        idx2 = mod(k+d-1, N) + 1;
        v1 = contour(idx1,:) - contour(k,:);
        v2 = contour(idx2,:) - contour(k,:);
        cosTheta = dot(v1,v2) / (norm(v1)*norm(v2) + eps);
        cosTheta = max(-1, min(1, cosTheta));
        angles(k) = acos(cosTheta);
    end
end
