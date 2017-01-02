function [out1, out2, out3] = gibbs_sampling(Y, Pi, P, W, C, n_it)

    T = size(Y,2);
    [M,K] = size(Pi);
    D = size(W,1);

    out1 = zeros(M*K,T); % <S_t^m>
    out2 = zeros(M*K,M*K,T); % <S_t^m S_t^n>
    out3 = zeros(M*K,K,T-1); % <S_t^m S_{t-1}^m>

    s = zeros(M*K, T);
    for t=1:T
        s(:,t) = reshape(mnrnd(1,ones(M,K)/K)',M*K,1);
    end


    % sacrificed iterations for burn-in
    burn_in     = 0;
    % consecutive samples are not independent e.g. use every 100th sample for estimation
    step_sample = 1;
    
    for it=1:(burn_in + (n_it * step_sample))
        
        for t = 1:T
            for m=1:M
                if t == 1
                    pa = (s((m-1)*K+1:m*K,t+1)'>0.5) * P((m-1)*K+1:m*K,:);
                    pb = Pi(m,:);
                elseif t == T
                    pa = ones(1,K);
                    pb = (s((m-1)*K+1:m*K, t-1)'>0.5) * P((m-1)*K+1:m*K,:);
                else
                    pa = (s((m-1)*K+1:m*K,t+1)'>0.5) * P((m-1)*K+1:m*K,:);
                    pb = (s((m-1)*K+1:m*K, t-1)'>0.5) * P((m-1)*K+1:m*K,:);
                end
                sformu = s(:,t);
                py = zeros(1,2);
                for k=1:K
                    sformu((m-1)*K+1:m*K,1) = (1:K==k)';
                    mu = sum(W'.*repmat(sformu,1,D),1);
                    py(1,k) = -1/2*(Y(:,t)'-mu)*pinv(C)*(Y(:,t)-mu')...
                        -D/2*log(2*pi)-0.5*log(det(C));
                end
                p = log(pa) + log(pb) + py;
                a = max(p);
                p = exp(p - (a + log(sum(exp(p-a)))));
                p = p / sum(p);
                s((m-1)*K+1:m*K,t) = mnrnd(1,p)';
                assert(sum(s((m-1)*K+1:m*K,t)) == 1);
            end
        end
        
        if it > burn_in
            if mod(it,step_sample) == 0
                out1 = out1 + s;
                for t=1:T
                    temp = s(:,t) * s(:,t)';
                    for m = 1:M
                        temp((m-1)*K+1:m*K,(m-1)*K+1:m*K) = diag(s((m-1)*K+1:m*K,t));
                        if t < T
                            out3((m-1)*K+1:m*K,:,t) = out3((m-1)*K+1:m*K,:,t) + ...
                            s((m-1)*K+1:m*K,t) * s((m-1)*K+1:m*K,t+1)';           
                        end
                    end
                    out2(:,:,t) = out2(:,:,t) + temp;
                end
            end
        end
    end
    
    out1 = out1 ./ n_it;
    out2 = out2 ./ n_it;
    out3 = out3 ./ n_it;

end
