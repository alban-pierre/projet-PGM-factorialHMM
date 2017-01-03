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
repeat = 1;

% Comparison performance
M = 3;
K = 2;
%[Y,Ytest,Pi,P,W,C] = generate_fhmm(T,K,M,D);
P1 = rand(M*K,K);
P2 = rand(M*K,K);
P3 = rand(M*K,K);
P4 = rand(M*K,K);
P1 = P1 ./ sum(P1,2);
P2 = P2 ./ sum(P2,2);
P3 = P3 ./ sum(P3,2);
P4 = P4 ./ sum(P4,2);

alls = eye(K);
for m=2:M
    alls = reshape(repmat(alls,1,K)',K*(m-1),K^m)';
    alls = [alls, repmat(eye(K), size(alls,1)/K,1)];
end


for t=[11,16,20]

    if (isMatlab)
        rng(t);
    else
        randn('seed',t); % 7, 11, 16, 20
    end
    [Y,Ytest,Pi,P,W,C] = generate_fhmm(T,K,M,D);
    
    % Different types of initialisations of W based on kmeans
    % If you use theses init put the recursive_kmeans_init in comment
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

    % Different types of initialisations of P
    if (false)
        P0 = rand(M*K,K);
    else % fhmm success : 20/20 (with the 3rd init of W doing 15/20)
        Y_2 = Y;
        ki = W0(:, 1:K);
        dd = sqdist(ki,Y_2);
        [~,imin] = min(dd,[],1);
        P0(1:K,:) = repmat(mean(repmat(imin,K,1) == repmat((1:K)',1,T),2)',K,1);
        for m=2:M
            for k=1:K
                Y_2 = Y_2 - repmat((imin == k),D,1).*repmat(ki(:,k),1,T);
            end
            ki = W0(:, (m-1)*K+1:m*K);
            dd = sqdist(ki,Y_2);
            [~,imin] = min(dd,[],1);
            P0((m-1)*K+1:m*K,:) = repmat(mean(repmat(imin,K,1) == repmat((1:K)',1,T),2)',K,1);
        end
    end

    C0 = diag(diag(cov(Y')));

    % The following line erase all previous init
    %[W0, P0, C0] = recursive_kmeans_init(Y, M, K);
    P0 = P0 ./ sum(P0,2);
    % End of initialisations
    
    % Exec
    [W1,C1,P1,Pi1,ll1] = em_fhmm(Y,K,M,maxIter,epsilon,W0,P0,C0);
    [W2,C2,P2,Pi2,ll2] = em_gibbs(Y,K,M,maxIter,epsilon,W0,P0,C0);
    [W3,C3,P3,Pi3,ll3] = em_cfva(Y,K,M,maxIter,epsilon,W0,P0,C0);
    try     % To avoid errors
        [W4,C4,P4,Pi4,ll4] = em_sva(Y,K,M,maxIter,epsilon,W0,P0,C0);
    catch
        fprintf('Error sva : M = %d, K = %d, rep = %d\n',M,K,t);
        continue
    end
    
    mu = W*alls';
    mu0 = W0*alls';
    mu1 = W1*alls';
    mu2 = W2*alls';
    mu3 = W3*alls';
    mu4 = W4*alls';


    % compute probabilities of each center
    p = zeros(M,K);
    for m=1:M
        [v,d] = eig(P((m-1)*K+1:m*K,:)');
        p(m,:) = v(:,(abs(diag(d) - 1) < 0.000001)');
    end
    p = p ./ sum(p,2);  
    p = repmat(reshape(p',K*M,1), 1, K^M).*alls';
    p = prod(reshape(p(alls'>0.5),M,K^M),1);
    
    p0 = zeros(M,K);
    for m=1:M
        [v,d] = eig(P0((m-1)*K+1:m*K,:)');
        p0(m,:) = v(:,(abs(diag(d) - 1) < 0.000001)');
    end
    p0 = p0 ./ sum(p0,2);  
    p0 = repmat(reshape(p0',K*M,1), 1, K^M).*alls';
    p0 = prod(reshape(p0(alls'>0.5),M,K^M),1);
    
    p1 = zeros(M,K);
    for m=1:M
        [v,d] = eig(P1((m-1)*K+1:m*K,:)');
        p1(m,:) = v(:,(abs(diag(d) - 1) < 0.000001)');
    end
    p1 = p1 ./ sum(p1,2);
    p1 = repmat(reshape(p1',K*M,1), 1, K^M).*alls';
    p1 = prod(reshape(p1(alls'>0.5),M,K^M),1);

    p2 = zeros(M,K);
    for m=1:M
        [v,d] = eig(P2((m-1)*K+1:m*K,:)');
        p2(m,:) = v(:,(abs(diag(d) - 1) < 0.000001)');
    end
    p2 = p2 ./ sum(p2,2);
    p2 = repmat(reshape(p2',K*M,1), 1, K^M).*alls';
    p2 = prod(reshape(p2(alls'>0.5),M,K^M),1);

    p3 = zeros(M,K);
    for m=1:M
        [v,d] = eig(P3((m-1)*K+1:m*K,:)');
        p3(m,:) = v(:,(abs(diag(d) - 1) < 0.000001)');
    end
    p3 = p3 ./ sum(p3,2);
    p3 = repmat(reshape(p3',K*M,1), 1, K^M).*alls';
    p3 = prod(reshape(p3(alls'>0.5),M,K^M),1);

    p4 = zeros(M,K);
    for m=1:M
        [v,d] = eig(P4((m-1)*K+1:m*K,:)');
        p4(m,:) = v(:,(abs(diag(d) - 1) < 0.000001)');
    end
    p4 = p4 ./ sum(p4,2);
    p4 = repmat(reshape(p4',K*M,1), 1, K^M).*alls';
    p4 = prod(reshape(p4(alls'>0.5),M,K^M),1);
    

    figure(t); hold off;
    plot(Y(1,:), Y(2,:), '.b');
    hold on;

    for i=1:K^M
        plot(mu(1,i), mu(2,i), 'bo', 'MarkerSize',10+round(p(1,i)*200),'LineWidth',1);
    end
    for i=1:K^M
        plot(mu0(1,i), mu0(2,i), 'co', 'MarkerSize',10+round(p0(1,i)*200),'LineWidth',1);
    end
    for i=1:K^M
        plot(mu1(1,i), mu1(2,i), 'ko', 'MarkerSize',10+round(p1(1,i)*200),'LineWidth',1);
    end
    for i=1:K^M
        plot(mu2(1,i), mu2(2,i), 'go', 'MarkerSize',10+round(p2(1,i)*200),'LineWidth',1);
    end
    for i=1:K^M
        plot(mu3(1,i), mu3(2,i), 'ro', 'MarkerSize',10+round(p3(1,i)*200),'LineWidth',1);
    end
    for i=1:K^M
        plot(mu4(1,i), mu4(2,i), 'mo', 'MarkerSize',10+round(p4(1,i)*200),'LineWidth',1);
    end
    
    plot(mu(1,:), mu(2,:), 'bx', 'MarkerSize',15,'LineWidth',3);
    plot(mu0(1,:), mu0(2,:), 'cx', 'MarkerSize',15,'LineWidth',3);
    plot(mu1(1,:), mu1(2,:), 'kx', 'MarkerSize',15,'LineWidth',3);
    plot(mu2(1,:), mu2(2,:), 'gx', 'MarkerSize',15,'LineWidth',3);
    plot(mu3(1,:), mu3(2,:), 'rx', 'MarkerSize',15,'LineWidth',3);
    plot(mu4(1,:), mu4(2,:), 'mx', 'MarkerSize',15,'LineWidth',3);

end
