function ll = loglikelihood(Y,W,C,P,Pi,states)
    [M,K] = size(Pi);
    
    if nargin < 6
        states = get_all_states(M,K);
    end
    
    Ptrans = computePtrans(P,states);
    mu = computeMu(W,states);
    gauss = computeGaussian(Y,mu,C);
    
    % Compute Pstates (t=1) for logAlpha1
    Pstates = ones(1,K^M);
    for i=1:K^M
        for m=1:M
            Pstates(i) = Pstates(i) * Pi(m,states(i,m));
        end
    end
    
    logAlpha1 = log(Pstates) + log(gauss(1,:));
    logBeta = betaRecursion(Pi,Ptrans,gauss);
    ab = max(logAlpha1 + logBeta(1,:),[],2);
    ll = ab + log(sum(exp(logAlpha1 + logBeta(1,:) - ab),2));
end