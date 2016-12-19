function logAlpha = alphaRecursion(Y,Pi,P,W,C)
    
    [D,T] = size(Y);
    [M,K] = size(Pi);
    logAlpha = zeros(T,K^M);
    mu = zeros(K^M,D);
    P_y = zeros(K^M,1);
    Pstates = ones(K^M,1);
    Ptrans=ones(K^M,K^M);
    
    % Compute mu, Pstates
    states = get_all_states(M,K);
    for i=1:K^M
        for m=1:M
            mu(i,:) = mu(i,:) + W(:,(m-1)*K+states(i,m))';
            Pstates(i) = Pstates(i) * Pi(m,states(i,m));
        end
        for j=1:K^M
            for m=1:M
                Ptrans(i,j)=Ptrans(i,j)*P(states(j,m),(m-1)*K+states(i,m));
            end
        end
    end
    
    for t = 0:T-1
        for i = 1:K^M
            P_y(i) = mvnpdf(Y(:,t+1)',mu(i,:),C);
        end
        
        if t == 0
            % Initialization
            logAlpha(1,:) = log(Pstates) + log(P_y);
        else
            % Recursion
            a = max(logAlpha(t,:));
            logAlpha(t+1,:) = log(P_y)' + a + log(exp(logAlpha(t,:)-a)*Ptrans'); 
        end
    end
    
end
