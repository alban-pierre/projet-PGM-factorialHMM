function [W,C,P,Pi,LL] = em_gibbs(Y,K,M,maxIter,epsilon)

    % Initialization
    [D,T] = size(Y);
    Pi = 1/K*ones(M,K);
    C = eye(D);
    W = randn(D,M*K);
    P = rand(M*K,K);
    P = P ./ sum(P,2);
    LL = [];
    
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
        
        % Log-likelihood
        L = rand(); % L = approx_loglikelihood(...);
        LL = [LL, L];
        
        % M step
        Pi = reshape(ESt(:,1)',[K,M])';
        W = sum1 * pinv(sum2);
        C = Y*Y'/T - 1/T * sum1 * W';
        C = (C+C')/2; % Make sure C is symmetric because of small computations errors
        P = sum3 ./ sum(sum3,2);
    
        if (tau > 1) && (LL(end) - LL(end-1) < epsilon)
            break;
        end
    end

end

