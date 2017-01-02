% For this file the dataset "CalIt2 Building People
% Counts Data Set" from UCI Repository is needed
% The files "Data2/data.csv" must exist

% It fails miserably

% Variable that is 1 if we use matlab, and 0 otherwise
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;

% To be able to repeat
if (isMatlab)
    rng('default');
    rng(1);
else
    pkg load statistics;
    randn('seed',2);
    rand('seed',2);
end


d = load('Data2/data.csv');
days = size(d,1)/96; % 105

Y = reshape(d(1:96*20), 2, 20*48);
Y_test = reshape(d(96*20+(1:96*10)), 2, 10*48);
Y_test2 = reshape(d(96*30+(1:96*10)), 2, 10*48);


K = 2;
M = 3;
maxIter = 100;
epsilon = 1e-4;


[W0, P0, C0] = recursive_kmeans_init(Y, M, K); % sometimes it fails because too many points are exactly the same
[W,C,P,Pi,ll] = em_fhmm(Y,K,M,maxIter,epsilon,W0,P0,C0);

LL = loglikelihood(Y_test,W,C,P,Pi)
LL2 = loglikelihood(Y_test2,W,C,P,Pi)
