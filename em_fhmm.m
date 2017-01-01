function [W,C,P,Pi,LL,time] = em_fhmm(Y,K,M,maxIter,epsilon,W0,P0)
    
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
    time = [];
    
    % Compute states
    states = get_all_states(M,K);
    
    % Build auxilary matrix to compute expectations
    aux=zeros(K^M,M*K); 
    aux2=zeros(K^M,M*M*K*K);
    for i=1:K^M
        for m=1:M
            aux(i,(m-1)*K+states(i,m)) = 1;
            for l=1:M
                aux2(i,((m-1)*K+states(i,m)-1)*M*K+(l-1)*K+states(i,l)) = 1;
            end
        end
    end
    
    for tau=1:maxIter
        tic
    
        % Compute Ptrans, mu and gauss 
        Ptrans = computePtrans(P,states);
        mu = computeMu(W,states);
        gauss = computeGaussian(Y,mu,C);
    
        % E step
        % Here we use alpha and beta recursions of an HMM of K^M states, and not
        % recursions of a factorial HMM (described in appendix B, equations B.1-4)
        % It gave the same results but it is slower for big values of K,M
            % You're right
        logAlpha = alphaRecursion(Pi,Ptrans,states,gauss);
        logBeta = betaRecursion(Pi,Ptrans,gauss);
        gamma = Gamma(logAlpha,logBeta);
        
        % <S_t>
        ESt = gamma * aux;
        
        % \sum_{t=1}^T Y_t <S_t>
        sum1 = zeros(D,K*M);
        for t = 1:T
            sum1 = sum1 + Y(:,t) * ESt(t,:);
        end
        
        % \sum_{t=1}^T <S_t S_t'>
        sum2=zeros(K*M,K*M);
        temp=gamma*aux2;
        for t=1:T
            temp1 = reshape(temp(t,:),K*M,K*M);
            for m = 1:M
                temp1((m-1)*K+1:m*K,(m-1)*K+1:m*K) = diag(ESt(t,(m-1)*K+1:m*K));
            end
            sum2 = sum2 + temp1;
        end
        
        % \sum_{t=2}^T <S_t^m S_{t-1}^m>
        sum3 = zeros(M*K,K);
        for t = 1:T-1
            for m = 1:M
                a = max(logAlpha(t,:));
                tempAlpha = a + log(exp(logAlpha(t,:)-a)*aux(:,(m-1)*K+1:m*K)); 
                b = max(logBeta(t+1,:)+log(gauss(t+1,:)));
                tempBeta = b + log(exp(logBeta(t+1,:)+log(gauss(t+1,:))-b)*aux(:,(m-1)*K+1:m*K)); 
                temp = log(P((m-1)*K+1:m*K,:)) + (tempAlpha' + tempBeta);
                c = max(temp(:));
                tempSum = c + log(sum(exp(temp(:)-c))); 
                sum3((m-1)*K+1:m*K,:) = sum3((m-1)*K+1:m*K,:) + exp(temp-tempSum);
            end  
        end
        
        % Log-likelihood
        ab = max(logAlpha(1,:) + logBeta(1,:),[],2);
        LL = [LL ab + log(sum(exp(logAlpha(1,:) + logBeta(1,:) - ab),2))];
        
        % M step
        Pi = reshape(ESt(1,:),[K,M])';
        W = sum1 * pinv(sum2);
        C = Y*Y'/T - 1/T * sum1 * W';
        C = (C+C')/2; % Make sure C is symmetric because of small computations errors
        P = sum3 ./ sum(sum3,2);
    
        time = [time , toc];
    
        if (tau > 1) && (LL(end) - LL(end-1) < epsilon)
            break;
        end
    end

end
