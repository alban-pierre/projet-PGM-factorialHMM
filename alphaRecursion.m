function logAlpha = alphaRecursion(Y,Pi,Ptrans,states,gauss)
    
    T = size(Y,2);
    [M,K] = size(Pi);
    logAlpha = zeros(T,K^M);
    Pstates = ones(1,K^M);
    
    % Compute Pstates
    states = get_all_states(M,K);
    for i=1:K^M
        for m=1:M
            Pstates(i) = Pstates(i) * Pi(m,states(i,m));
        end
    end
    
    for t = 0:T-1
    	if t == 0
            % Initialization
            logAlpha(1,:) = log(Pstates) + log(gauss(t+1,:));
        else
            % Recursion
            a = max(logAlpha(t,:));
	        logAlpha(t+1,:) = log(gauss(t+1,:)) + a + log(exp(logAlpha(t,:)-a)*Ptrans); 
        end
    end
    
end
