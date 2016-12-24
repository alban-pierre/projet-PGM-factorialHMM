function gauss = computeGaussian(Y,mu,C) 
    T = size(Y,2);
    KpM = size(mu,1);
    gauss = zeros(T,KpM);

    for i = 1:KpM
        gauss(:,i) = mvnpdf(Y',mu(i,:),C);
    end
end