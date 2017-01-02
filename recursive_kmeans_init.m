function [W0, P0, C0] = recursive_kmeans_init(Y, M, K)

    [D,T] = size(Y);

    W0 = zeros(D,K*M);
    Y_2 = Y;
    for m=1:M
        ki = kmeans(Y_2, K, 0.01);
        W0(:, (m-1)*K+1:m*K) = ki;
        dd = sqdist(ki,Y_2);
        [~,imin] = min(dd,[],1);
        P0((m-1)*K+1:m*K,:) = repmat(mean(repmat(imin,K,1) == repmat((1:K)',1,T),2)',K,1);  
        for k=1:K
            Y_2 = Y_2 - repmat((imin == k),D,1).*repmat(ki(:,k),1,T);
        end
    end

    P0 = P0 ./ sum(P0,2);

    C0 = mean(eig(cov(Y_2')));

end
