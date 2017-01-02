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
D = 2;
T = 100;
maxIter = 100;
epsilon = 1e-4;
repeat = 3;

LL0 = []; LL0test = [];
LL1f = []; LL1testf = [];
LL2f = []; LL2testf = [];
LL3f = []; LL3testf = [];
LL4f = []; LL4testf = [];

% Comparison performance
M = 3;
K = 2;
[Y,Ytest,Pi,P,W,C] = generate_fhmm(T,K,M,D);

for t=1:repeat

    % Init
    W0 = randn(D,M*K);
    P0 = rand(M*K,K);
    P0 = P0 ./ sum(P0,2);
    
    % Exec
    [W1,C1,P1,Pi1,ll1] = em_fhmm(Y,K,M,maxIter,epsilon,W0,P0);
    [W2,C2,P2,Pi2,ll2] = em_gibbs(Y,K,M,maxIter,epsilon,W0,P0);
    [W3,C3,P3,Pi3,ll3] = em_cfva(Y,K,M,maxIter,epsilon,W0,P0);
    try     % To avoid errors
        [W4,C4,P4,Pi4,ll4] = em_sva(Y,K,M,maxIter,epsilon,W0,P0);
    catch
        fprintf('Error sva : M = %d, K = %d, rep = %d\n',M,K,t);
        continue
    end
    
    figure(1); hold off;
    plot(Y(1,:), Y(2,:), '.b');
    hold on;

    alls = eye(K);
    for m=2:M
        alls = reshape(repmat(alls,1,K)',K*(m-1),K^m)';
        alls = [alls, repmat(eye(K), size(alls,1)/K,1)];
    end

    mu = W*alls';
    mu1 = W1*alls';
    mu2 = W2*alls';
    mu3 = W3*alls';
    mu4 = W4*alls';


    figure(t); hold off;
    plot(Y(1,:), Y(2,:), '.b');
    hold on;
    plot(mu(1,:), mu(2,:), 'bx', 'MarkerSize',15,'LineWidth',3);
    plot(mu1(1,:), mu1(2,:), 'kx', 'MarkerSize',15,'LineWidth',3);
    plot(mu2(1,:), mu2(2,:), 'gx', 'MarkerSize',15,'LineWidth',3);
    plot(mu3(1,:), mu3(2,:), 'rx', 'MarkerSize',15,'LineWidth',3);
    plot(mu4(1,:), mu4(2,:), 'mx', 'MarkerSize',15,'LineWidth',3);

end
