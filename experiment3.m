% For this file the dataset "Activity Recognition from Single Chest-Mounted
% Accelerometer Data Set" from UCI Repository is needed
% The files "Data/1.csv" must exist

% It fails miserably

% Variable that is 1 if we use matlab, and 0 otherwise
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;

% To be able to repeat
if (isMatlab)
    rng('default');
    rng(1);
else
    pkg load statistics;
    randn('seed',8);
    rand('seed',8);
end


% Collect 7 types of data of size T=100, D=3
d = load('Data/1.csv');

b = 1;
T = 100;
i = d(b,5);
ndata = ones(7,1);
clear data;
for n=1:size(d,1)
    if n == b+T-1
        data{i, ndata(i,1)} = d(b:n,2:4)';
        ndata(i,1) = ndata(i,1) + 1;
        b = n+1;
    end
    if d(n,5) ~= i
        b = n;
        i = d(b,5);
    end
end

ndata = ndata-1;

M = 3;
K = 2;

LL = zeros(3,3);
clear Ytest;
for i=1:3
    Ytest{i} = data{i,9};
end
% Training and testing over 7 examples each
for i=1:3
    [W0, P0, C0] = recursive_kmeans_init(Y, M, K);
    [W,C,P,Pi,ll] = em_fhmm(data{i,1},K,M,maxIter,epsilon,W0,P0,C0);

    for j=1:3
        LL(i,j) = loglikelihood(Ytest{j},W,C,P,Pi);
    end
end

%plot(1:7, LL');
[~, imax] = max(LL, [], 1);
imax == 1:3
