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
repeat = 20;

LL0 = []; LL0test = [];
LL1f = []; LL1testf = [];
LL2f = []; LL2testf = [];
LL3f = []; LL3testf = [];
LL4f = []; LL4testf = [];

% Comparison performance
M = 3;
K = 2;
[Y,Ytest,Pi,P,W,C] = generate_fhmm(T,K,M,D);

alls = eye(K);
for m=2:M
    alls = reshape(repmat(alls,1,K)',K*(m-1),K^m)';
    alls = [alls, repmat(eye(K), size(alls,1)/K,1)];
end


for t=1:repeat

    if (isMatlab)
        rng(t);
    else
        randn('seed',t);
    end
    [Y,Ytest,Pi,P,W,C] = generate_fhmm(T,K,M,D);
    
    % Different types of initialisations based on kmeans
    if (false) % fhmm success : 5/20
        W0 = randn(D,M*K);
    elseif (false) % fhmm success : 6/20
        ki = kmeans(Y, K^M, 0.01);
        sd = reshape(repmat(alls',1,D)',K^M,K*M*D);
        W0 = sd .* repmat(ki', 1, K*M);
        W0 = reshape(K/M*mean(W0,1),D,K*M);
    elseif (true) % fhmm success : 15/20
        W0 = zeros(D,K*M);
        Y_2 = Y;
        ki = kmeans(Y_2, K, 0.01);
        W0(:, 1:K) = ki;
        for m=2:M
            dd = sqdist(ki,Y_2);
            [~,imin] = min(dd,[],1);
            for k=1:K
                Y_2 = Y_2 - repmat((imin == k),D,1).*repmat(ki(:,k),1,T);
            end
            ki = kmeans(Y_2, K, 0.01);
            W0(:, (m-1)*K+1:m*K) = ki;
        end
    elseif (false) % fhmm success : 10/20
        W0 = zeros(D,K*M);
        Y_2 = kmeans(Y, K^M, 0.01);
        ki = kmeans(Y_2, K, 0.01);
        W0(:, 1:K) = ki;
        for m=2:M
            dd = sqdist(ki,Y_2);
            [~,imin] = min(dd,[],1);
            for k=1:K
                Y_2 = Y_2 - repmat((imin == k),D,1).*repmat(ki(:,k),1,K^M);
            end
            ki = kmeans(Y_2, K, 0.01);
            W0(:, (m-1)*K+1:m*K) = ki;
        end
    else % fhmm success : 10/20
        ni = 10;
        score = zeros(1,ni);
        clear W0;
        for i=1:ni
            W0{i} = zeros(D,K*M);
            Y_2 = Y;
            ki = kmeans(Y_2, K, 0.01);
            W0{i}(:, 1:K) = ki;
            for m=2:M
                dd = sqdist(ki,Y_2);
                [~,imin] = min(dd,[],1);
                for k=1:K
                    Y_2 = Y_2 - repmat((imin == k),D,1).*repmat(ki(:,k),1,T);
                end
                ki = kmeans(Y_2, K, 0.01);
                W0{i}(:, (m-1)*K+1:m*K) = ki;
            end
            mu0 = W0{i}*alls';
            dd = sqdist(mu0,Y);
            score(1,i) = sum(min(dd,[],1),2);
        end
        [~, imin] = min(score,[],2);
        W0 = W0{imin};
    end

    P0 = rand(M*K,K);
    P0 = P0 ./ sum(P0,2);
    % End of initialisations
    
    % Exec
    [W1,C1,P1,Pi1,ll1] = em_fhmm(Y,K,M,maxIter,epsilon,W0,P0);
    %[W2,C2,P2,Pi2,ll2] = em_gibbs(Y,K,M,maxIter,epsilon,W0,P0);
    %[W3,C3,P3,Pi3,ll3] = em_cfva(Y,K,M,maxIter,epsilon,W0,P0);
    try     % To avoid errors
        %[W4,C4,P4,Pi4,ll4] = em_sva(Y,K,M,maxIter,epsilon,W0,P0);
    catch
        fprintf('Error sva : M = %d, K = %d, rep = %d\n',M,K,t);
        continue
    end
    
    mu = W*alls';
    mu0 = W0*alls';
    mu1 = W1*alls';
    %mu2 = W2*alls';
    %mu3 = W3*alls';
    %mu4 = W4*alls';


    figure(t); hold off;
    plot(Y(1,:), Y(2,:), '.b');
    hold on;
    plot(mu(1,:), mu(2,:), 'bx', 'MarkerSize',15,'LineWidth',3);
    plot(mu0(1,:), mu0(2,:), 'cx', 'MarkerSize',15,'LineWidth',3);
    plot(mu1(1,:), mu1(2,:), 'kx', 'MarkerSize',15,'LineWidth',3);
    %plot(mu2(1,:), mu2(2,:), 'gx', 'MarkerSize',15,'LineWidth',3);
    %plot(mu3(1,:), mu3(2,:), 'rx', 'MarkerSize',15,'LineWidth',3);
    %plot(mu4(1,:), mu4(2,:), 'mx', 'MarkerSize',15,'LineWidth',3);

end
