function [out1, out2, out3, s] = gibbs_sampling(Y, Pi, P, W, C, n_it, s)

    T = size(Y,2);
    [M,K] = size(Pi);
    D = size(W,1);

    out1 = zeros(M*K,T); % <S_t^m>
    out2 = zeros(M*K,M*K,T); % <S_t^m S_t^n>
    out3 = zeros(M*K,K,T-1); % <S_t^m S_{t-1}^m>

    % If possible we reuse the last sampling
    if (nargin < 7)
        s = zeros(M*K, T);
        for t=1:T
            s(:,t) = reshape(mnrnd(1,ones(M,K)/K)',M*K,1);
        end
    end

    % They don't burn iterations in their experimentation, so you can delete it if you want
    % sacrificed iterations for burn-in
    burn_in     = 5;
    % consecutive samples are not independent e.g. use every 100th sample for estimation
    step_sample = 5;
    
    for it=1:(burn_in + (n_it * step_sample))
        
        for t = 1:T
            for m=1:M
                if t == 1
                    pa = P((m-1)*K+1:m*K,s((m-1)*K+1:m*K,t+1)'>0.5);
                    pb = Pi(m,:)';
                elseif t == T
                    pa = ones(K,1);
                    pb = P((m-1)*K+1:m*K,:);
                    pb = pb(s((m-1)*K+1:m*K, t-1)>0.5, :)';
                else
                    pa = P((m-1)*K+1:m*K,s((m-1)*K+1:m*K,t+1)'>0.5);
                    pb = P((m-1)*K+1:m*K,:);
                    pb = pb(s((m-1)*K+1:m*K, t-1)>0.5, :)';
                end
                sformu = s(:,t);
                for k=1:K
                    sformu((m-1)*K+1:m*K,1) = (1:K==k)';
                    mu = sum(W'.*repmat(sformu,1,D),1);
                    py(k,:) = mvnpdf(Y(:,t)',mu,C);
                end
                % The following four lines avoid probabilities like p = [1, 10e-54], but there
                % must be a better way to do this
                if (it < burn_in/2)
                    py = py./(10*max(max(py,1),[],1));
                    py = 1./(-log(py));
                end
                p = pa.*pb.*py;
                p = p'/sum(p,1);
                s((m-1)*K+1:m*K,t) = mnrnd(1,p)';
                assert(sum(s((m-1)*K+1:m*K,t)) == 1);
            end
        end
        
        if it > burn_in
            if mod(it,step_sample) == 0
                out1 = out1 + s;
                for t=1:T
                    out2(:,:,t) = out2(:,:,t) + repmat(s(:,t),1,K*M).*repmat(s(:,t),1,K*M)';
                    
                    if t < T
                        out3(:,:,t) = out3(:,:,t) + repmat(s(:,t+1), 1,K) .* ...
                            reshape(repmat(reshape(s(:,t),K,M)',1,K)',K,K*M)';
                    end
                end
            end
        end
    end
    
    out1 = out1 ./ n_it;
    out2 = out2 ./ n_it;
    out3 = out3 ./ n_it;

end
