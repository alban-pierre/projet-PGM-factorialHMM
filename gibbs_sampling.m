function [out1, out2, out3] = gibbs_sampling(Y, Pi, Ptrans, W, C, n_it)

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

    
    for it=1:n_it
        
        %first node
        t = 1;
        for m=1:M
            pa = P((m-1)*K+1:m*K,s((m-1)*K+1:m*K,t+1)'>0.5);
            pb = ones(K,1);
            sformu = s(:,t);
            for k=1:K
                sformu((m-1)*K+1:m*K,1) = (1:K==k)';
                mu = sum(W'.*repmat(sformu,1,D),1);
                py(k,:) = mvnpdf(Y(:,t)',mu,C);
            end
            py = py./(10*max(max(py,1),[],1));
            py = 1./(-log(py));
            p = pa.*pb.*py;
            p = p'/sum(p,1);
            s((m-1)*K+1:m*K,t) = mnrnd(1,p)';
        end
        
        %nodes in the middle
        for t = 2:T-1
            for m=1:M
                pa = P((m-1)*K+1:m*K,s((m-1)*K+1:m*K,t+1)'>0.5);
                pb = P((m-1)*K+1:m*K,:);
                pb = pb(s((m-1)*K+1:m*K, t-1)>0.5, :)';
                sformu = s(:,t);
                for k=1:K
                    sformu((m-1)*K+1:m*K,1) = (1:K==k)';
                    mu = sum(W'.*repmat(sformu,1,D),1);
                    py(k,:) = mvnpdf(Y(:,t)',mu,C);
                end
                py = py./(10*max(max(py,1),[],1));
                py = 1./(-log(py));
                p = pa.*pb.*py;
                p = p'/sum(p,1);
                s((m-1)*K+1:m*K,t) = mnrnd(1,p)';
            end
        end
        
        %last node
        t = T;
        for m=1:M
            pa = ones(K,1);
            pb = P((m-1)*K+1:m*K,:);
            pb = pb(s((m-1)*K+1:m*K, t-1)>0.5, :)';
            sformu = s(:,t);
            for k=1:K
                sformu((m-1)*K+1:m*K,1) = (1:K==k)';
                mu = sum(W'.*repmat(sformu,1,D),1);
                py(k,:) = mvnpdf(Y(:,t)',mu,C);
            end
            py = py./(10*max(max(py,1),[],1));
            py = 1./(-log(py));
            p = pa.*pb.*py;
            p = p'/sum(p,1);
            s((m-1)*K+1:m*K,t) = mnrnd(1,p)';
        end

        out1 = out1 + s;
        for t=1:T;
            out2(:,:,t) = out2(:,:,t) + repmat(s(:,t),1,K*M).*repmat(s(:,t),1,K*M)';
        end
        for t=1:T-1
            out3(:,:,t) = out3(:,:,t) + repmat(s(:,t+1), 1,K) .* reshape(repmat(reshape(s(:,t),K,M)',1,K)',K,K*M)';
        end
        
    end
    out1 = out1 ./ n_it;
    out2 = out2 ./ n_it;
    out3 = out3 ./ n_it;
end
