function mu = computeMu(W,states)
    D = size(W,1);
    [KpM,M] = size(states);
    K = size(W,2) / M;
    mu = zeros(KpM,D);
    
    for i=1:KpM
        for m=1:M
            mu(i,:) = mu(i,:) + W(:,(m-1)*K+states(i,m))';
        end
    end
end