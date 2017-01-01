function [Y,Ytest,Pi,P,W,C] = generate_fhmm(T,K,M,D)

    % Generate random parameters

    Pi = rand(M,K);
    Pi = Pi ./ sum(Pi,2);

    P = rand(M*K,K);
    P = P ./ sum(P,2);

    W = randn(D,M*K);

    %C = diag(rand(1,D)); % To be sure it is psd
    C = 0.01 * eye(D);

    % Compute sequence of hidden and observable varibles
    S = zeros(M*K,T);
    Y = zeros(D,T);
    Stest = zeros(M*K,T);
    Ytest = zeros(D,T);
    
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

end
