% EM algorithm with structural variational approximation

function [W,C,P,Pi,LL] = em_sva(Y,K,M,maxIter,epsilon)
    
    % Initialization
    [D,T] = size(Y);
    Pi = 1/K*ones(M,K);
    C = eye(D);
    W = randn(D,M*K);
    P = rand(M*K,K);
    P = P ./ sum(P,2);
    LL = [];
    aLL = [];

    ESt = zeros(T,M*K);
    h = ones(T,M*K);
    
    states = get_all_states(M,K);
    states1 = get_all_states(1,K);
    
    for tau=1:maxIter        
        % E Step
        
        invC = pinv(C);
        
        % Computation of theta
        for i = 1:10 % For convergence of KL divergence
            % Compute Delta
            Delta = zeros(K,M);
            for m = 1:M
                Delta(:,m) = diag(W(:,(m-1)*K+1:m*K)' * ...
                    invC * W(:,(m-1)*K+1:m*K));
            end
        
            hNew = zeros(T,M*K);
            
            for t = 1:T
                for m = 1:M
                    % Compute Ytilda
                    ind = 1:M;
                    ind(m) = [];
                    temp = 0;
                    for l = ind
                        temp = temp + W(:,(l-1)*K+1:l*K) * ...
                            ESt(t,(l-1)*K+1:l*K)';
                    end
                    Ytilda = Y(:,t) - temp;
                    
                    % Compute hNew
                    hNew(t,(m-1)*K+1:m*K) = exp(W(:,(m-1)*K+1:m*K)' * ...
                        invC * Ytilda - 0.5 * Delta(:,m));
                end
            end
            
            h = hNew;
        end

        % Alpha/Beta recursions on each HMM
        logAlpha = zeros(T,K,M);
        logBeta = zeros(T,K,M);
        gamma = zeros(T,K,M);
        for m=1:M
            logAlpha(:,:,m) = alphaRecursion(Pi(m,:),P((m-1)*K+1:m*K,:),states1,h(:,(m-1)*K+1:m*K));
            logBeta(:,:,m) = betaRecursion(Pi(m,:),P((m-1)*K+1:m*K,:),h(:,(m-1)*K+1:m*K));
            gamma(:,:,m) = Gamma(logAlpha(:,:,m),logBeta(:,:,m));
        end

        ESt = reshape(gamma, T, K*M);
        
        sum1 = zeros(D,K*M); % \sum_{t=1}^T Y_t <S_t>
        sum2 = zeros(K*M,K*M); % \sum_{t=1}^T <S_t^m S_t^n'>
        sum3 = zeros(M*K,K); % \sum_{t=2}^T <S_t^m S_{t-1}^m>
        for t = 1:T          
            sum1 = sum1 + Y(:,t) * ESt(t,:);
            
            temp = ESt(t,:)' * ESt(t,:);
            for m = 1:M
                temp((m-1)*K+1:m*K,(m-1)*K+1:m*K) = diag(ESt(t,(m-1)*K+1:m*K));
                
                if t < T
                    sum3((m-1)*K+1:m*K,:) = sum3((m-1)*K+1:m*K,:) + ...
                        ESt(t,(m-1)*K+1:m*K)' * ESt(t+1,(m-1)*K+1:m*K);           
                end
            end
            sum2 = sum2 + temp;
        end
        
        % Approx log-likelihood
        aLL = [aLL approx_loglikelihood_cfva(Y,ESt,W,invC,P,Pi)];
        
        % True log-likelihood
        LL = [LL , loglikelihood(Y,W,C,P,Pi,states)];
        
        % M step
        Pi = reshape(ESt(1,:),[K,M])';
        W = sum1 * pinv(sum2);
        C = Y*Y'/T - 1/T * sum1 * W';
        C = (C+C')/2; % Make sure C is symmetric because of small computations errors
        P = sum3 ./ sum(sum3,2);
        
        % Break if convergence
        if (tau > 1) && (aLL(end) - aLL(end-1) < epsilon)
            break;
        end
    end
end
