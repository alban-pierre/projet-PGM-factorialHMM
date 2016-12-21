%%% NOT FINISHED

function [W,C,P,Pi,LL] = em_fhmm(Y,K,M,maxIter,tol)
    
    % Initialization
    [D,T] = size(Y);
    Pi = 1/K*ones(M,K);
    C = eye(D);
    W = randn(D,M*K);
    P = rand(K,M*K);
    P = P ./ sum(P,2);
    LL = [];
    
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
        %%% E step
        logAlpha = alphaRecursion(Y,Pi,P,W,C);
        logBeta = betaRecursion(Y,Pi,P,W,C);
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
        
        % Log-likelihood
        ab = max(logAlpha(1,:) + logBeta(1,:),[],2);
        LL = [LL ab + log(sum(exp(logAlpha(1,:) + logBeta(1,:) - ab),2))];
        
        % M step
        Pi = reshape(ESt(1,:)*aux,[K,M])';
        W = sum1 * pinv(sum2);
        %C = Y*Y'/T - 1/T * ;
        
        if (tau > 1) && (LL(end) - LL(end-1) < tol)
            break;
        end
    end

end
