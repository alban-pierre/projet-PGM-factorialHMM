function [W,C,P,Pi,LL] = em_gibbs(Y,K,M,maxIter,epsilon)

    % Initialization
    [D,T] = size(Y);
    Pi = 1/K*ones(M,K);
    C = eye(D);
    W = randn(D,M*K);
    P = rand(M*K,K);
    P = P ./ sum(P,2);
    LL = [];
    aLL = [];
    states = get_all_states(M,K);
    
    %initialisation of gibbs_sampling
    s = zeros(M*K, T);
    for t=1:T
        s(:,t) = reshape(mnrnd(1,ones(M,K)/K)',M*K,1);
    end
    

    for tau=1:maxIter

        % E step
        [ESt, ESmSn, EStSt, s] = gibbs_sampling(Y, Pi, P, W, C, 10, s);
        
        % \sum_{t=1}^T Y_t <S_t>
        sum1 = zeros(D,K*M);
        for t = 1:T
            sum1 = sum1 + Y(:,t) * ESt(:,t)';
        end
        
        % \sum_{t=1}^T <S_t S_t'>
        sum2=sum(ESmSn, 3);
        
        % \sum_{t=2}^T <S_t^m S_{t-1}^m>
        sum3 = sum(EStSt, 3);
        
        % Approx log-likelihood
        invC = pinv(C);
        aLL = [aLL, approx_loglikelihood_gibbs(ESt,ESmSn,EStSt,Y,W,invC,P,Pi)];
        
        % True log-likelihood
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
        LL = [LL ab + log(sum(exp(logAlpha1 + logBeta(1,:) - ab),2))];
        
        % M step
        Pi = reshape(ESt(:,1)',[K,M])';
        W = sum1 * pinv(sum2);
        C = Y*Y'/T - 1/T * sum1 * W';
        C = (C+C')/2; % Make sure C is symmetric because of small computations errors
        P = sum3 ./ sum(sum3,2);
    
        if (tau > 1) && (aLL(end) - aLL(end-1) < epsilon)
            break;
        end
    end

end

