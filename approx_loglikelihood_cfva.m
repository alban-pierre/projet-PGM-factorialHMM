% cf formula A.1 
function ll = approx_loglikelihood_cfva(Y,theta,W,invC,P,Pi)
    [D,T] = size(Y);
    [M,K] = size(Pi);
    
    ll = 0;
    
    for t = 1:T
        ll = ll - 0.5 * Y(:,t)' * invC * Y(:,t);
        for m = 1:M
            ll = ll + Y(:,t)' * invC * W(:,(m-1)*K+1:m*K) * ...
                theta(t,(m-1)*K+1:m*K)';
            for n = 1:M
                if m==n
                    ll = ll - 0.5 * trace(W(:,(m-1)*K+1:m*K)' * invC *...
                        W(:,(m-1)*K+1:m*K) * diag(theta(t,(m-1)*K+1:m*K)));
                else
                    ll = ll -0.5 * trace(W(:,(m-1)*K+1:m*K)' * invC *...
                        W(:,(n-1)*K+1:n*K) * theta(t,(n-1)*K+1:n*K)' * ...
                        theta(t,(m-1)*K+1:m*K));
                end
            end
            if t < T
                ll = ll + trace(log(P((m-1)*K+1:m*K,:)) * ...
                    theta(t,(m-1)*K+1:m*K)' * theta(t+1,(m-1)*K+1:m*K));
            end
            % Normalization term
            ll = ll - sum(theta(t,(m-1)*K+1:m*K) .* log(theta(t,(m-1)*K+1:m*K)));
        end
    end
    
    temp = Pi';
    temp = temp(:);
    ll = ll + theta(1,:) * log(temp);
    
    % Normalization term
    ll = ll + T * log((2*pi)^(-D/2) * sqrt(det(invC)));
    
end
