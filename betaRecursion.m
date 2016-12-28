function logBeta = betaRecursion(Pi,Ptrans,gauss)
    
    %T = size(Y,2);
    T = size(gauss,1);
    [M,K] = size(Pi);
    logBeta = zeros(T,K^M);
    
    for t = T:-1:2
        b = max(logBeta(t,:));
        logBeta(t-1,:) = b + log((gauss(t,:).*exp(logBeta(t,:)-b))*Ptrans'); 
    end
    
end
