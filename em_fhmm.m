%%% NOT FINISHED

function [W,C,P,Pi,LL] = em_fhmm(Y,K,M,maxIter,tol)
    
    % Initialization
    [D,T] = size(Y);
    Pi = 1/K*ones(M,K);
    C = eye(D);
    W = randn(D,M*K);
    P = rand(K,M*K);
    P = P ./ sum(P);
    LL = [];
    
    states = get_all_states(M,K);
    
    % Build auxilary matrix to compute expectations
    aux=zeros(K^M,M*K); 
    for i=1:K^M
        for m=1:M
            aux(i,(m-1)*K+states(i,m)) = 1;
        end
    end
    
    for tau=1:maxIter
        %%% E step
        logAlpha = alphaRecursion(Y,Pi,P,W,C);
        logBeta = betaRecursion(Y,Pi,P,W,C);
        gamma = Gamma(logAlpha,logBeta);
        
        % Log-likelihood
        ab = max(logAlpha(1,:) + logBeta(1,:),[],2);
        LL = [LL ab + log(sum(exp(logAlpha(1,:) + logBeta(1,:) - ab),2))];
        
        % M step
        Pi = reshape(gamma(1,:)*aux,[K,M])';
        W = 
        %C = Y*Y'/T - 1/T * ;
        
        if (tau > 1) && (LL(end) - LL(end-1) < tol)
            break;
        end
    end

end
