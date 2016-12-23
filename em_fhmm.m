function [W,C,P,Pi,LL] = em_fhmm(Y,K,M,maxIter,epsilon)
    
    % Initialization
    [D,T] = size(Y);
    Pi = 1/K*ones(M,K);
    C = eye(D);
    W = randn(D,M*K);
    P = rand(M*K,K);
    P = P ./ sum(P,2);
    LL = [];
    
    % Compute states and Ptrans
    states = get_all_states(M,K);
    Ptrans = computePtrans(P,states);
    
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
    	% Compute mu and gauss 
        mu = computeMu(W,states);
        gauss = computeGaussian(Y,mu,C);
	
        % E step
        logAlpha = alphaRecursion(Y,Pi,Ptrans,states,gauss);
        logBeta = betaRecursion(Y,Pi,Ptrans,gauss);
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
            sum2 = sum2 + reshape(temp(t,:),K*M,K*M);
        end
	
	% \sum_{t=2}^T <S_t^m S_{t-1}^m>
        sum3 = zeros(M*K,K);
        for t = 1:T-1 
            for m = 1:M
                temp = log(P((m-1)*K+1:m*K,:)) + ((logAlpha(t,:) * aux(:,(m-1)*K+1:m*K))' * ...
                       ((logBeta(t+1,:)+log(gauss(t+1,:))) * aux(:,(m-1)*K+1:m*K)));
                sum3((m-1)*K+1:m*K,:) = sum3((m-1)*K+1:m*K,:) + exp(temp/sum(sum(temp)));        
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
        if (tau > 1) && (LL(end) - LL(end-1) < epsilon)
            break;
        end
    end

end
