clc;clear;
g_zebra(:,:,1) = get_matrix('Zebra_gt1.raw',481,321,0); 
g_zebra(:,:,2) = get_matrix('Zebra_gt2.raw',481,321,0);
g_zebra(:,:,3) = get_matrix('Zebra_gt3.raw',481,321,0);
g_zebra(:,:,4) = get_matrix('Zebra_gt4.raw',481,321,0);
g_zebra(:,:,5) = get_matrix('Zebra_gt5.raw',481,321,0);
g_jaguar(:,:,1) = get_matrix('Jaguar_gt1.raw',481,321,0); 
g_jaguar(:,:,2) = get_matrix('Jaguar_gt2.raw',481,321,0);
g_jaguar(:,:,3) = get_matrix('Jaguar_gt3.raw',481,321,0);
g_jaguar(:,:,4) = get_matrix('Jaguar_gt4.raw',481,321,0);
g_jaguar(:,:,5) = get_matrix('Jaguar_gt5.raw',481,321,0);
g_jaguar(:,:,6) = get_matrix('Jaguar_gt6.raw',481,321,0);

% filenameinput = input('Enter the file name');
image1 = get_matrix('Zebra.raw',481,321,1);
% Performing Canny edge detection
canny_edges1 = edge(rgb2gray(image1),'Canny',[5/255.0 100/255.0]);
figure
imshow(1-canny_edges1);
imwrite(1-canny_edges1,'Canny_zebra_7.png')


image2 = get_matrix('Jaguar.raw',481,321,1);
% Performing Canny edge detection
canny_edges2 = edge(rgb2gray(image2),'Canny',[100/255.0 150/255.0]);
figure
imshow(1-canny_edges2);
imwrite(1-canny_edges2,'Canny_jaguar_7.png')

% Finding F1 score
height = 481;
width = 321;

Ground_truth = struct('groundTruth',[]);
Ground_truth.groundTruth = cell(1,1);
for i=1:1
Ground_truth.groundTruth{i} = struct('Boundaries',[]);
end
%zebra
for i = 1:5
    Ground_truth.groundTruth{1}.Boundaries = 1 - g_zebra(:,:,i)./255;

    [thrs,cntR,sumR,cntP,sumP,V]=edgesEvalImg(1-canny_edges1, Ground_truth);
    R1 = cntR./sumR;
    P1 = cntP./(sumP + 0.0001); 
    mean_P(i) = mean(P1)
    mean_R(i) = mean(R1)
end

mean_P_final = mean(mean_P)
mean_R_final = mean(mean_R)
F = (2 * mean_R_final * mean_P_final)/ (mean_R_final + mean_P_final );
disp(F)


% jaguar
for i = 1:6
    Ground_truth.groundTruth{1}.Boundaries = 1 - g_jaguar(:,:,i)./255;
    
    [thrs,cntR,sumR,cntP,sumP,V]=edgesEvalImg(1-canny_edges1, Ground_truth);
    R2 = cntR./sumR;
    P2 = cntP./(sumP + 0.0001); 
    mean_P(i) = mean(P2);
    mean_R(i) = mean(R2);
end

mean_P_final = mean(mean_P)
mean_R_final = mean(mean_R)
F = (2 * mean_R_final * mean_P_final)/ (mean_R_final + mean_P_final );
disp(F)