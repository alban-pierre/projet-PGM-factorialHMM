function Ptrans = computePtrans(P,states)
    [KpM,M] = size(states);
    K = size(P,2);
    Ptrans=ones(KpM,KpM);
    
    for i=1:KpM
        for j=1:KpM
            for m=1:M
                Ptrans(i,j)=Ptrans(i,j)*P((m-1)*K+states(i,m),states(j,m));
            end
        end
    end
end