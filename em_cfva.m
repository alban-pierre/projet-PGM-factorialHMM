% EM algorithm with completely factorized variational approximation

function [W,C,P,Pi,LL] = em_cfva(Y,K,M,maxIter,epsilon)
    
    % Initialization
    [D,T] = size(Y);
    Pi = 1/K*ones(M,K);
    C = eye(D);
    W = randn(D,M*K);
    P = rand(M*K,K);
    P = P ./ sum(P,2);
    LL = [];
    
    theta = ones(T,M*K) / K;
    
    for tau=1:maxIter        
        % E Step
        % Computation of theta
        for i = 1:10 % For convergence of KL divergence
            % Compute Delta
            Delta = zeros(K,M);
            for m = 1:M
                Delta(:,m) = diag(W(:,(m-1)*K+1:m*K)' * ...
                    pinv(C)*W(:,(m-1)*K+1:m*K));
            end
        
            % Compute Ytilda
            Ytilda = zeros(T,D*M);
            for t = 1:T
                for m = 1:M
                    ind = 1:M;
                    ind(m) = [];
                    temp = 0;
                    for l = ind
                        temp = temp + W(:,(l-1)*K+1:l*K) * ...
                            theta(t,(l-1)*K+1:l*K)';
                    end
                    Ytilda(t,(m-1)*D+1:m*D) = Y(:,t) - temp;
                end
            end
        
            % Compute theta
            thetaNew = zeros(T,M*K);
            % t = 1
            for m = 1:M
                thetaNew(1,(m-1)*K+1:m*K) = softmax(W(:,(m-1)*K+1:m*K)' * ...
                    pinv(C) * Ytilda(1,(m-1)*D+1:m*D)' - 0.5 * Delta(:,m) + ...
                    log(P((m-1)*K+1:m*K,:))' * theta(2,(m-1)*K+1:m*K)');
            end
        
            for t = 2:T-1
                for m = 1:M
                    thetaNew(t,(m-1)*K+1:m*K) = softmax(W(:,(m-1)*K+1:m*K)' * ...
                        pinv(C) * Ytilda(t,(m-1)*D+1:m*D)' - 0.5 * Delta(:,m) + ...
                        log(P((m-1)*K+1:m*K,:)) * theta(t-1,(m-1)*K+1:m*K)' + ...
                        log(P((m-1)*K+1:m*K,:))' * theta(t+1,(m-1)*K+1:m*K)');
                end
            end
        
            % t = T
            for m = 1:M
                thetaNew(T,(m-1)*K+1:m*K) = softmax(W(:,(m-1)*K+1:m*K)' * ...
                    pinv(C) * Ytilda(1,(m-1)*D+1:m*D)' - 0.5 * Delta(:,m) + ...
                    log(P((m-1)*K+1:m*K,:)) * theta(T-1,(m-1)*K+1:m*K)');
            end
            
            theta = thetaNew;
        end
        
        % \sum_{t=1}^T Y_t <S_t>
        sum1 = zeros(D,K*M);
        for t = 1:T
            sum1 = sum1 + Y(:,t) * theta(t,:);
        end
        
        % \sum_{t=1}^T <S_t^m S_t^n'>
        sum2 = zeros(K*M,K*M);
        for t=1:T
            temp = theta(t,:)' * theta(t,:);
            for m = 1:M
                temp((m-1)*K+1:m*K,(m-1)*K+1:m*K) = diag(theta(t,(m-1)*K+1:m*K));
            end
            sum2 = sum2 + temp;
        end
        
        % \sum_{t=2}^T <S_t^m S_{t-1}^m>
        sum3 = zeros(M*K,K);
        for t = 1:T-1 
            for m = 1:M
                sum3((m-1)*K+1:m*K,:) = sum3((m-1)*K+1:m*K,:) + ...
                    theta(t,(m-1)*K+1:m*K)' * theta(t+1,(m-1)*K+1:m*K);        
            end  
        end
        
        % Log-likelihood
        % ...
        
        % M step
        Pi = reshape(theta(1,:),[K,M])';
        W = sum1 * pinv(sum2);
        C = Y*Y'/T - 1/T * sum1 * W';
        C = (C+C')/2; % Make sure C is symmetric because of small computations errors
        P = sum3 ./ sum(sum3,2);
        
        % Break if convergence
        % ...
    end