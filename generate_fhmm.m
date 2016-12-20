%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y = [Y_1,...,Y_T]                         D  * T
% S_t = [S^{1}|...|S^{M}]'                  MK * 1
% S = [S_1,...,S_T]                         MK * T
% W = [W^{1}|...|W^{M}]                     D  * MK
% P_{i,j} = P(S_{t+1} = i | S_{t+j} = j) 
% P = [P^{1}|...|P^{M}]]'                   MK  * K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear;

% To be able to repeat
rng('default');
rng(1);

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

W = randn(D,M*K);

%C = diag(rand(1,D)); % To be sure it is psd
C = 0.1 * eye(D);

% Compute sequence of hidden and observable varibles
S = zeros(M*K,T);
Y = zeros(D,T);

% Initialization
S(:,1) = reshape(mnrnd(1,Pi)',[K*M,1]);
mu = 0;
for m = 1:M
    mu = mu + W(:,((m-1)*K+1):m*K) * S((m-1)*K+1:m*K,1);
end
Y(:,1) = mvnrnd(mu,C);
    
for t = 2:T
    mu = 0;
    for m = 1:M
        S((m-1)*K+1:m*K,t) = reshape(mnrnd(1,S((m-1)*K+1:m*K,t-1)'*P((m-1)*K+1:m*K,:)),[K,1]);
        mu = mu + W(:,((m-1)*K+1):m*K) * S((m-1)*K+1:m*K,t);
    end
    Y(:,t) = mvnrnd(mu,C);
end

plot(Y(1,:),Y(2,:),'.');
