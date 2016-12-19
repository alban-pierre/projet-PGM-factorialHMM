function states = get_all_states(M,K)
    V = 1:K;
    
    [temp{1:M}] = ndgrid(V);
    states = reshape(cat(M+1,temp{:}),[],M); 
    
end