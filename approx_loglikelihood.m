% cf formula A.1 
function ll = approx_loglikelihood(Y,theta,W,C,P,Pi)
    [D,T] = size(Y);
    [M,K] = size(Pi);
    
    ll = 0;
    
    for t = 1:T
        ll = ll - 0.5 * Y(:,t)'*pinv(C)*Y(:,t);
        for m = 1:M
            ll = ll + Y(:,t)' * pinv(C)*W(:,(m-1)*K+1:m*K) * ...
                theta(t,(m-1)*K+1:m*K)';
            for n = 1:M
                if m==n
                    ll = ll - 0.5 * trace(W(:,(m-1)*K+1:m*K)' * pinv(C) *...
                        W(:,(m-1)*K+1:m*K) * diag(theta(t,(m-1)*K+1:m*K)));
                else
                    ll = ll -0.5 * trace(W(:,(m-1)*K+1:m*K)' * pinv(C) *...
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
    ll = ll + T * log((2*pi)^(-D/2)/sqrt(det(C)));
    
end