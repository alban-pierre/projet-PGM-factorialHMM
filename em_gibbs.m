function [W,C,P,Pi,LL,aLL,time] = em_gibbs(Y,K,M,maxIter,epsilon,W0,P0)

    [D,T] = size(Y);
    
    if nargin < 6
        W = randn(D,M*K);
        P = rand(M*K,K);
        P = P ./ sum(P,2);
    else
        W = W0;
        P = P0;
    end

    % Initialization
    Pi = 1/K*ones(M,K);
    C = diag(diag(cov(Y')));
    LL = [];
    aLL = [];
    time = [];
    states = get_all_states(M,K);
    
    for tau=1:maxIter
        tic

        % E step
        [ESt, ESmSn, EStSt] = gibbs_sampling(Y, Pi, P, W, C, 10);

        
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
        LL = [LL , loglikelihood(Y,W,C,P,Pi,states)];
        
        % M step
        Pi = reshape(ESt(:,1)',[K,M])';
        W = sum1 * pinv(sum2);
        C = Y*Y'/T - 1/T * sum1 * W';
        C = (C+C')/2; % Make sure C is symmetric because of small computations errors
        for i=1:K*M 
            temp=sum(sum3(i,:));
            if(temp==0)
                P(i,:) = ones(1,K) / K; % To avoid error
            else
                P(i,:) = sum3(i,:)/temp;
            end
        end
    
        time = [time , toc];
    
        if (tau > 1) && (aLL(end) - aLL(end-1) < epsilon)
            break;
        end
    end

end

