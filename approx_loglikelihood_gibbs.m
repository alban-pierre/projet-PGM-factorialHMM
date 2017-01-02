% cf formula A.1 
function ll = approx_loglikelihood_gibbs(ESt,ESmSn,EStSt,Y,W,invC,P,Pi)

    [D,T] = size(Y);
    [M,K] = size(Pi);
    
    ll = 0;
    
    for t = 1:T
        ll = ll - 0.5 * Y(:,t)' * invC * Y(:,t);
        for m = 1:M
            ll = ll + Y(:,t)' * invC * W(:,(m-1)*K+1:m*K) * ...
                ESt((m-1)*K+1:m*K,t);
            for n = 1:M
                ll = ll -0.5 * trace(W(:,(m-1)*K+1:m*K)' * invC *...
                    W(:,(n-1)*K+1:n*K) * ESmSn((n-1)*K+1:n*K,(m-1)*K+1:m*K,t));
            end
            if t < T
                ll = ll + trace(log(P((m-1)*K+1:m*K,:)) * ...
                    EStSt((m-1)*K+1:m*K,:,t));
            end
            % Normalization term
            %ll = ll - sum(theta(t,(m-1)*K+1:m*K) .* log(theta(t,(m-1)*K+1:m*K)));
        end
    end
    
    temp = Pi';
    temp = temp(:);
    ll = ll + ESt(:,1)' * log(temp);
    
    % Normalization term
    ll = ll + T * log((2*pi)^(-D/2) * sqrt(det(invC)));
    
end
