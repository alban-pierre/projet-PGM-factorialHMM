%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y = [Y_1,...,Y_T]                         D  * T
% S_t = [S^{1}|...|S^{M}]'                  MK * 1
% S = [S_1,...,S_T]                         MK * T
% W = [W^{1}|...|W^{M}]                     D  * MK
% P_{i,j} = P(S_{t+1} = i | S_{t+j} = j) 
% P = [P^{1}|...|P^{M}]]'                   MK  * K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear;

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

% Parameters
K = 2;
M = 3;
D = 2;
T = 500;

% Generate random parameters

Pi = rand(M,K);
Pi = Pi ./ sum(Pi,2);

P = rand(M*K,K);
P = P ./ sum(P,2);

W = 2 * randn(D,M*K);

%C = diag(rand(1,D)); % To be sure it is psd
C = 0.1 * eye(D);

% Compute sequence of hidden and observable varibles
S = zeros(M*K,T);
Y = zeros(D,T);

% Initialization
S(:,1) = reshape(mnrnd(1,Pi)',[K*M,1]);
mu = sum(W(:,S(:,1)>0),2);
Y(:,1) = mvnrnd(mu,C);
Stest(:,1) = reshape(mnrnd(1,Pi)',[K*M,1]);
muTest = sum(W(:,Stest(:,1)>0),2);
Ytest(:,1) = mvnrnd(muTest,C);

for t = 2:T
    S(:,t) = reshape(mnrnd(1,P(S(:,t-1)>0,:))', [K*M,1]);
    mu = sum(W(:,S(:,t)>0),2);
    Y(:,t) = mvnrnd(mu,C);
    Stest(:,t) = reshape(mnrnd(1,P(Stest(:,t-1)>0,:))', [K*M,1]);
    muTest = sum(W(:,Stest(:,t)>0),2);
    Ytest(:,t) = mvnrnd(muTest,C);
end

plot(Y(1,:),Y(2,:),'.');
