function gamma = Gamma(logAlpha,logBeta)
    
    % log P(y_1,...,y_T)
    ab = max(logAlpha + logBeta,[],2);
    logP_y = ab + log(sum(exp(logAlpha + logBeta - ab),2));
    
    gamma = exp(logAlpha + logBeta - logP_y);
    
end