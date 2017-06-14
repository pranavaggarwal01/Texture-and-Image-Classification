
clc;clear;
width = 481;
height = 321;
z = get_matrix('Zebra.raw',481,321,1);
g(:,:,1) = get_matrix('Zebra_gt1.raw',481,321,0); 
g(:,:,2) = get_matrix('Zebra_gt2.raw',481,321,0);
g(:,:,3) = get_matrix('Zebra_gt3.raw',481,321,0);
g(:,:,4) = get_matrix('Zebra_gt4.raw',481,321,0);
g(:,:,5) = get_matrix('Zebra_gt5.raw',481,321,0);
%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts);  % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=10;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results

tic, E=edgesDetect(z,model); toc
figure(1); imshow(z); figure(2); imshow(1-E);
imwrite(1-E,'SE_zebra_7.png')
% E = E ./ max(E(:));
% E(E < 0.5) = 1;
% E(E >= 0.5) = 0;
for i = 1:height
    for j = 1:width
        if E(i,j) < 0.25
           edge_image(i,j) = 0;
        else
           edge_image(i,j) = 255;
        end
    end
end
figure
imshow(255-edge_image);
imwrite(255-edge_image,'Thresholded_zebra_7.png')
% figure
% imshow(g);
Ground_truth = struct('groundTruth',[]);
Ground_truth.groundTruth = cell(1,1);
for i=1:1
Ground_truth.groundTruth{i} = struct('Boundaries',[]);
end
for i = 1:5
    Ground_truth.groundTruth{1}.Boundaries = 1 - g(:,:,i)./255;
    % Ground_truth.groundTruth{2}.Boundaries = 1 - g2./255;
    % Ground_truth.groundTruth{3}.Boundaries = 1 - g3./255;
    % Ground_truth.groundTruth{4}.Boundaries = 1 - g4./255;
    % Ground_truth.groundTruth{5}.Boundaries = 1 - g5./255;

    [thrs,cntR,sumR,cntP,sumP,V]=edgesEvalImg(edge_image./255, Ground_truth);
    R = cntR./sumR;
    P = cntP./(sumP + 0.0001); 
    mean_P(i) = mean(P);
    mean_R(i) = mean(R);
end

mean_P_final = mean(mean_P)
mean_R_final = mean(mean_R)
F = (2 * mean_R_final * mean_P_final)/ (mean_R_final + mean_P_final );
disp(F)
toc;
