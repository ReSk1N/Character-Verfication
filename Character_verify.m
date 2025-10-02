clear; close all; clc;
%%
train_images = cell(50, 1);
test_images  = cell(50, 1);

% Load train image
for n = 1:25
    train_images{n} = imread(sprintf("D:/專題/Character Verification/9/database/base_1_%d_9.bmp", n));
    train_images{n+25} = imread(sprintf("D:/專題/Character Verification/9/testcase/word_%d_1_9.bmp", n));
end

% Load test image
for n = 26:50
    test_images{n-25} = imread(sprintf("D:/專題/Character Verification/9/database/base_1_%d_9.bmp", n));
    test_images{n} = imread(sprintf("D:/專題/Character Verification/9/testcase/word_%d_1_9.bmp", n));
end

% Feature Extraction WITHOUT keypoints
train_features = zeros(50, 11);
test_features  = zeros(50, 11);

for i = 1:50
    train_features(i, :) = extract_all_features(train_images{i});

    test_features(i, :) = extract_all_features(test_images{i});
end

%% Feature Extraction for train keypoints
d = 15;     % step size
L = 15;     % phi kernel size (odd)
weight = 1; % weight for rank + theta terms
threshRatio = 1; % Filter stable points

% Choose a standard image and extract features
% std_img = imread("D:\專題\Character Verification\1\database\base_1_11_1.bmp");
% std_img = imread("D:\專題\Character Verification\2\database\base_1_22_2.bmp");
% std_img = imread("D:\專題\Character Verification\3\database\base_1_17_3.bmp");
% std_img = imread("D:\專題\Character Verification\4\database\base_1_22_4.bmp");
% std_img = imread("D:\專題\Character Verification\5\database\base_1_22_5.bmp");
% std_img = imread("D:\專題\Character Verification\6\database\base_1_10_6.bmp");
% std_img = imread("D:\專題\Character Verification\7\database\base_1_23_7.bmp");
% std_img = imread("D:\專題\Character Verification\8\database\base_1_10_8.bmp");
std_img = imread("D:\專題\Character Verification\9\database\base_1_40_9.bmp");
stdKP = extract_keypoint_features(std_img, d, L);

numStdEnd = size(stdKP.end_feat,1);
numStdTurn = size(stdKP.turn_feat,1);

end_count = zeros(numStdEnd,1);
turn_count = zeros(numStdTurn,1);

for n = 1:50
    train_img = train_images{n};
    trainKP = extract_keypoint_features(train_img, d, L);

    % Find endpoints to keep
    [end_matches, ~] = find_keep_points(trainKP.end_feat, stdKP.end_feat, weight);
    for i = 1:numel(end_matches)
        end_count(end_matches(i)) = end_count(end_matches(i)) + 1;
    end

    % Find turning points to keep
    [turn_matches, ~] = find_keep_points(trainKP.turn_feat, stdKP.turn_feat, weight);
    for i = 1:numel(turn_matches)
        turn_count(turn_matches(i)) = turn_count(turn_matches(i)) + 1;
    end
end

% FILTER STABLE KEYPOINTS
end_threshold = threshRatio * 50;
turn_threshold = threshRatio * 50;

stable_end_idx = find(end_count >= end_threshold);
stable_turn_idx = find(turn_count >= turn_threshold);

% Keep the stable points only
std_end_feat_stable = stdKP.end_feat(stable_end_idx,:);
std_turn_feat_stable = stdKP.turn_feat(stable_turn_idx,:);
std_end_xy_stable = stdKP.end_xy(stable_end_idx,:);
std_turn_xy_stable = stdKP.turn_xy(stable_turn_idx,:);

% Number of stable points (end + turn combined)
numStable = size(std_end_feat_stable,1) + size(std_turn_feat_stable,1);

% Allocate storage for coordinates: 50 × (2*numStable)
train_stable_coords = nan(50, 2*numStable);

% Get coordinate features
for n = 1:50
    train_img = train_images{n};
    trainKP = extract_keypoint_features(train_img, d, L);

    % Match stable endpoints 
    [stable_end_matches, ~] = match_points(std_end_feat_stable, trainKP.end_feat, weight);

    % Match stable turning points
    [stable_turn_matches, ~] = match_points(std_turn_feat_stable, trainKP.turn_feat, weight);

    % Combine matches (end first, then turn)
    matches = [stable_end_matches; stable_turn_matches];

    % Save coordinates into feature array
    for i = 1:numStable
        if i <= numel(stable_end_matches) % endpoint
            j = matches(i);
            if j > 0 && j <= size(trainKP.end_xy,1)
                train_stable_coords(n,i) = trainKP.end_xy(j,1);              % m coord
                train_stable_coords(n,i+numStable) = trainKP.end_xy(j,2);    % n coord
            end
        else % turning point
            j = matches(i);
            if j > 0 && j <= size(trainKP.turn_xy,1)
                train_stable_coords(n,i) = trainKP.turn_xy(j,1);             % m coord
                train_stable_coords(n,i+numStable) = trainKP.turn_xy(j,2);   % n coord
            end
        end
    end
end

% Get direction features
P = numStable; % number of stable points
numPairs = nchoosek(P,2);
numFeatures = numPairs * 4; % each pair contributes 4

train_direction_features = nan(50, numFeatures);

for n = 1:50
    % Extract stable coordinates for image n
    m_coords = train_stable_coords(n, 1:P);
    n_coords = train_stable_coords(n, P+1:end);

    feat_vec = [];
    % Loop over pairs (i < j)
    for i = 1:P-1
        for j = i+1:P
            dm = m_coords(j) - m_coords(i);
            dn = n_coords(j) - n_coords(i);

            % Add [Δm, Δn, -Δm, -Δn]
            feat_vec = [feat_vec, dm, dn, -dm, -dn]; %#ok<AGROW>
        end
    end
    train_direction_features(n,:) = feat_vec;
end

% Concatenate features
train_kp_features = [train_stable_coords, train_direction_features];
disp("Training features size:"); disp(size(train_kp_features));

%% Test image keypoint part
test_stable_coords = nan(50, 2*numStable);

for n = 1:50
    test_img = test_images{n};
    testKP = extract_keypoint_features(test_img, d, L);

    % Match stable endpoints
    [stable_end_matches, ~] = match_points(std_end_feat_stable, testKP.end_feat, weight);

    % Match stable turning points
    [stable_turn_matches, ~] = match_points(std_turn_feat_stable, testKP.turn_feat, weight);

    % Combine matches
    matches = [stable_end_matches; stable_turn_matches];

    % Save coordinates
    for i = 1:numStable
        if i <= numel(stable_end_matches) % endpoint
            j = matches(i);
            if j > 0 && j <= size(testKP.end_xy,1)
                test_stable_coords(n,i) = testKP.end_xy(j,1);
                test_stable_coords(n,i+numStable) = testKP.end_xy(j,2);
            end
        else % turning point
            j = matches(i);
            if j > 0 && j <= size(testKP.turn_xy,1)
                test_stable_coords(n,i) = testKP.turn_xy(j,1);
                test_stable_coords(n,i+numStable) = testKP.turn_xy(j,2);
            end
        end
    end
end

% Direction features
test_direction_features = nan(50, numFeatures);

for n = 1:50
    m_coords = test_stable_coords(n,1:P);
    n_coords = test_stable_coords(n,P+1:end);
    feat_vec = [];
    for i = 1:P-1
        for j = i+1:P
            dm = m_coords(j)-m_coords(i);
            dn = n_coords(j)-n_coords(i);
            feat_vec = [feat_vec, dm, dn, -dm, -dn]; %#ok<AGROW>
        end
    end
    test_direction_features(n,:) = feat_vec;
end

test_kp_features = [test_stable_coords, test_direction_features];
disp("Testing features size:"); disp(size(test_kp_features));

%% Concatenating + Normalizing
train_features = [train_features, train_kp_features];
test_features = [test_features, test_kp_features];

mean_train = mean(train_features, 1);
std_train = std(train_features, 0, 1);
std_train(std_train == 0) = 1;

norm_train = (train_features - mean_train) ./ std_train;
norm_test = (test_features - mean_train) ./ std_train;
%% SVM (Change folder to Character Verification\libsvm-3.36\matlab)
addpath("D:\專題\Character Verification\libsvm-3.36\matlab");

% Need MinGW add-on
make 
train_labels = [ones(25,1);zeros(25,1)];  % 25 true + 25 forged
test_labels = [ones(25,1);zeros(25,1)];   % 25 true + 25 forged

model = svmtrain(train_labels, norm_train, '-t 0 -c 1');
[predicted_labels, accuracy, dec_values] = svmpredict(test_labels, norm_test, model);

% disp(predicted_labels);

%% Functions
function features = extract_all_features(rgbimage)
    % Convert to binary
    rgbdouble = double(rgbimage);
    Y = 0.299 * rgbdouble(:,:,1) + 0.587 * rgbdouble(:,:,2) + 0.114 * rgbdouble(:,:,3);
    binaryimage = Y < 220;  % Thresholding

    % Stroke features (1–10)
    stroke_features = extract_stroke_features(binaryimage);

    % Moment features (11–19)
    [m0, n0] = compute_centroids(binaryimage);
    V20 = compute_moment_features(binaryimage, m0, n0, 2, 0);
    V11 = compute_moment_features(binaryimage, m0, n0, 1, 1);
    V02 = compute_moment_features(binaryimage, m0, n0, 0, 2);
    V30 = compute_moment_features(binaryimage, m0, n0, 3, 0);
    V21 = compute_moment_features(binaryimage, m0, n0, 2, 1);
    V12 = compute_moment_features(binaryimage, m0, n0, 1, 2);
    V03 = compute_moment_features(binaryimage, m0, n0, 0, 3);
    moment_features = [m0, n0, V20, V11, V02, V30, V21, V12, V03];

    % Intensity features (20–21) 
    intensity_features = extract_intensity_features(rgbimage);

    % Erosion features (22–24)
    erosion_features = extract_erosion_features(binaryimage);

    % Combining all features
    % features = [stroke_features, moment_features, intensity_features, erosion_features];
    features = [moment_features, intensity_features];
end

function stroke_features = extract_stroke_features(binaryimage) % Returns 1x10
    [rows, cols] = size(binaryimage);
    stroke_features = zeros(1, 10);

    width = floor(cols/5);
    height = floor(rows/5); 

    for k = 1:5 % f1 to f5
        col_start = (k-1)*width + 1;
        if k < 5
            col_end = k*width;
        else
            col_end = cols; 
        end
        segment = binaryimage(:, col_start:col_end);
        stroke_features(k) = sum(segment(:)); 
    end

    for k = 1:5 %f6 to f10
        row_start = (k-1)*height + 1;
        if k < 5
            row_end = k*height;
        else
            row_end = rows; 
        end
        segment = binaryimage(row_start:row_end, :);
        stroke_features(5+k) = sum(segment(:)); 
    end
end

function [m0, n0] = compute_centroids(B)
    [rows, cols] = size(B);
    B = double(B);
    [M, N] = ndgrid(1:rows, 1:cols);
    total = sum(B(:));
    if total == 0
        m0 = 0; n0 = 0;
    else
        m0 = sum(sum(M .* B)) / total;
        n0 = sum(sum(N .* B)) / total;
    end
end

function Vab = compute_moment_features(B, m0, n0, a, b)
    [rows, cols] = size(B);
    B = double(B);
    [M, N] = ndgrid(1:rows, 1:cols);
    total = sum(B(:));
    if total == 0
        Vab = 0;
    else
        Vab = sum(sum(((M - m0).^a) .* ((N - n0).^b) .* B)) / total;
    end
end

function int_feats = extract_intensity_features(rgbimage)
    rgbdouble = double(rgbimage);
    Y = 0.299 * rgbdouble(:,:,1) + 0.587 * rgbdouble(:,:,2) + 0.114 * rgbdouble(:,:,3);
    meanY = mean(Y(:));
    stdY  = std(Y(:));
    int_feats = [meanY, stdY];
end

function erosion_feats = extract_erosion_features(binaryimage)
    total_pixels = sum(binaryimage(:));
    erosion_feats = zeros(1,3);
    BW = binaryimage;
    for k = 1:3
        BW = Erode(BW, 1);  % Apply one erosion at a time
        erosion_feats(k) = sum(BW(:)) / total_pixels;
    end
end

function output = Erode(A, iter)
    output = A;
    for k = 1:iter
        padded = padarray(output, [1 1], Inf);
        for m = 2:size(padded,1)-1
            for n = 2:size(padded,2)-1
                output(m-1,n-1) = min([
                    padded(m,n), ...
                    padded(m-1,n), ...
                    padded(m+1,n), ...
                    padded(m,n-1), ...
                    padded(m,n+1)
                ]);
            end
        end
    end
end

%% Functions for keypoint features
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
                contour = [contour; r_curr, c_curr]; %#ok<AGROW>
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
                            contour = [contour; r_next, c_next]; %#ok<AGROW>
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
                contours{end+1} = contour - 1; %#ok<AGROW>
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
            if theta_k <= pi/6 % endpoint
                endpoints(end+1) = k; %#ok<AGROW>

            elseif theta_k < 5*pi/6 % turning point
                turningpoints(end+1) = k; %#ok<AGROW>
            end
        end
    end
end


function angles = computeContourAngles(contour, d) % For classifying
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

function KP = extract_keypoint_features(imgRGB, d, L) % Extracts coordinate, rank, direction
    rgbdouble = double(imgRGB);
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

    [m_all,n_all] = find(B);
    m_all = double(m_all); n_all = double(n_all);
    m0 = (max(m_all)+min(m_all))/2;
    n0 = (max(n_all)+min(n_all))/2;
    w = max(max(m_all)-min(m_all), max(n_all)-min(n_all)); if w==0, w=1; end
    mr_fun = @(m) mean(m_all<m);
    nr_fun = @(n) mean(n_all<n);

    phi = build_phi_kernel(L);
    im_phi = conv2(double(B), phi, 'same');
    theta_map = angle(im_phi)/pi;

    end_feat = zeros(size(end_xy_all,1),5);
    for i=1:size(end_xy_all,1)
        m = double(end_xy_all(i,1)); n = double(end_xy_all(i,2));
        m_head = 2*(m-m0)/w; n_head = 2*(n-n0)/w;
        end_feat(i,:) = [m_head, n_head, mr_fun(m), nr_fun(n), theta_map(round(m),round(n))];
    end

    turn_feat = zeros(size(turn_xy_all,1),5);
    for i=1:size(turn_xy_all,1)
        m = double(turn_xy_all(i,1)); n = double(turn_xy_all(i,2));
        m_head = 2*(m-m0)/w; n_head = 2*(n-n0)/w;
        turn_feat(i,:) = [m_head, n_head, mr_fun(m), nr_fun(n), theta_map(round(m),round(n))];
    end

    KP.end_feat = end_feat; KP.turn_feat = turn_feat;
    KP.end_xy = end_xy_all; KP.turn_xy = turn_xy_all;
end

function phi = build_phi_kernel(L) % For direction feature
    assert(mod(L,2)==1,'L must be odd');
    L1=(L-1)/2;
    [N,M]=meshgrid(-L1:L1,-L1:L1);
    denom = sqrt(M.^2+N.^2);
    phi = zeros(L,L);
    mask=denom>0;
    phi(mask) = (-1j*M(mask)+N(mask))./denom(mask);
end

% Find points that appear in at least 80% of traning images
function [match_idx, d2_vals] = find_keep_points(inp_feat,std_feat,weight) 
    if isempty(inp_feat)||isempty(std_feat)
        match_idx=[]; d2_vals=[]; return;
    end
    Ni=size(inp_feat,1); 
    match_idx=zeros(Ni,1); d2_vals=zeros(Ni,1);
    for i=1:Ni
        dm=(inp_feat(i,1)-std_feat(:,1)).^2;
        dn=(inp_feat(i,2)-std_feat(:,2)).^2;
        dmr=(inp_feat(i,3)-std_feat(:,3)).^2;
        dnr=(inp_feat(i,4)-std_feat(:,4)).^2;
        dtheta_raw=abs(inp_feat(i,5)-std_feat(:,5));
        dtheta_wrap=min(dtheta_raw,2-dtheta_raw);
        d2=dm+dn+weight*(dmr+dnr)+weight*dtheta_wrap;
        [d2_min,j]=min(d2);
        match_idx(i)=j; d2_vals(i)=d2_min;
    end
end

% Match stable points in standard image to all images in the set
function [match_idx, d2_vals] = match_points(std_feat, inp_feat, weight)
    if isempty(std_feat) || isempty(inp_feat)
        match_idx=[]; d2_vals=[]; return;
    end
    Ns = size(std_feat,1);
    
    match_idx = zeros(Ns,1);
    d2_vals   = zeros(Ns,1);
    for i = 1:Ns
        dm  = (std_feat(i,1)-inp_feat(:,1)).^2;
        dn  = (std_feat(i,2)-inp_feat(:,2)).^2;
        dmr = (std_feat(i,3)-inp_feat(:,3)).^2;
        dnr = (std_feat(i,4)-inp_feat(:,4)).^2;
        dtheta_raw  = abs(std_feat(i,5)-inp_feat(:,5));
        dtheta_wrap = min(dtheta_raw, 2-dtheta_raw);
        d2 = dm + dn + weight*(dmr+dnr) + weight*dtheta_wrap;
        [d2_min, j] = min(d2);
        match_idx(i) = j; d2_vals(i) = d2_min;
    end
end