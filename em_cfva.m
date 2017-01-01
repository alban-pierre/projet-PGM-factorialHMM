% EM algorithm with completely factorized variational approximation

function [W,C,P,Pi,LL,aLL,time] = em_cfva(Y,K,M,maxIter,epsilon,W0,P0)
    
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
    
    theta = ones(T,M*K) / K;
    
    states = get_all_states(M,K);
    
    for tau=1:maxIter     
        tic
    
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
        
            thetaNew = zeros(T,M*K);
            
            for t = 1:T
                for m = 1:M
                    % Compute Ytilda
                    ind = 1:M;
                    ind(m) = [];
                    temp = 0;
                    for l = ind
                        temp = temp + W(:,(l-1)*K+1:l*K) * ...
                            theta(t,(l-1)*K+1:l*K)';
                    end
                    Ytilda = Y(:,t) - temp;
                    
                    % Compute thetaNew
                    if t == 1
                        thetaNew(1,(m-1)*K+1:m*K) = softmax(W(:,(m-1)*K+1:m*K)' * ...
                            invC * Ytilda - 0.5 * Delta(:,m) + log(Pi(m,:))' + ...
                            log(P((m-1)*K+1:m*K,:))' * theta(2,(m-1)*K+1:m*K)');
                    elseif t == T
                        thetaNew(T,(m-1)*K+1:m*K) = softmax(W(:,(m-1)*K+1:m*K)' * ...
                            invC * Ytilda - 0.5 * Delta(:,m) + ...
                            log(P((m-1)*K+1:m*K,:)) * theta(T-1,(m-1)*K+1:m*K)');
                    else
                        thetaNew(t,(m-1)*K+1:m*K) = softmax(W(:,(m-1)*K+1:m*K)' * ...
                            invC * Ytilda - 0.5 * Delta(:,m) + ...
                            log(P((m-1)*K+1:m*K,:)) * theta(t-1,(m-1)*K+1:m*K)' + ...
                            log(P((m-1)*K+1:m*K,:))' * theta(t+1,(m-1)*K+1:m*K)');
                    end
                end
            end
            
            theta = thetaNew;
        end
        
        sum1 = zeros(D,K*M); % \sum_{t=1}^T Y_t <S_t>
        sum2 = zeros(K*M,K*M); % \sum_{t=1}^T <S_t^m S_t^n'>
        sum3 = zeros(M*K,K); % \sum_{t=2}^T <S_t^m S_{t-1}^m>
        for t = 1:T          
            sum1 = sum1 + Y(:,t) * theta(t,:);
            
            temp = theta(t,:)' * theta(t,:);
            for m = 1:M
                temp((m-1)*K+1:m*K,(m-1)*K+1:m*K) = diag(theta(t,(m-1)*K+1:m*K));
                
                if t < T
                    sum3((m-1)*K+1:m*K,:) = sum3((m-1)*K+1:m*K,:) + ...
                        theta(t,(m-1)*K+1:m*K)' * theta(t+1,(m-1)*K+1:m*K);           
                end
            end
            sum2 = sum2 + temp;
        end
        
        % Approx log-likelihood
        aLL = [aLL approx_loglikelihood_cfva(Y,theta,W,invC,P,Pi)];
        % Not sure if I'm using the right loglikelihood
        
        % True log-likelihood
        LL = [LL , loglikelihood(Y,W,C,P,Pi,states)];
        
        % M step
        Pi = reshape(theta(1,:),[K,M])';
        W = sum1 * pinv(sum2);
        C = Y*Y'/T - 1/T * sum1 * W';
        C = (C+C')/2; % Make sure C is symmetric because of small computations errors
        P = sum3 ./ sum(sum3,2);
        
        time = [time , toc];
        
        % Break if convergence
        if (tau > 1) && (aLL(end) - aLL(end-1) < epsilon)
            break;
        end
    end
end
