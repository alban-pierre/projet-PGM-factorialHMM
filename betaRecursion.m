function logBeta = betaRecursion(Y,Pi,P,W,C)
    
    [D,T] = size(Y);
    [M,K] = size(Pi);
    logBeta = zeros(T,K^M);
    mu = zeros(K^M,D);
    P_y = zeros(K^M,1);
    Ptrans=ones(K^M,K^M);
    
    % Compute mu, Ptrans
    states = get_all_states(M,K);
    for i=1:K^M
        for m=1:M
            mu(i,:) = mu(i,:) + W(:,(m-1)*K+states(i,m))';
        end
        for j=1:K^M
            for m=1:M
                Ptrans(i,j)=Ptrans(i,j)*P((m-1)*K+states(i,m),states(j,m));
            end
        end
    end
    
    for t = T:-1:2
        for i = 1:K^M
            P_y(i) = mvnpdf(Y(:,t)',mu(i,:),C);
        end
        
        b = max(logBeta(t,:));
        logBeta(t-1,:) = b + log(P_y'.*exp((logBeta(t,:)-b))*Ptrans'); 
 
    end
    
end
