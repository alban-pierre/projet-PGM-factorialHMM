%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y = [Y_1,...,Y_T]                         D  * T
% S_t = [S^{1}|...|S^{M}]'                  MK * 1
% S = [S_1,...,S_T]                         MK * T
% W = [W^{1}|...|W^{M}]                     D  * MK
% P_{k,l} = P(S_{t+1} = l | S_{t+j} = k) 
% P = [P^{1}|...|P^{M}]]'                   MK * K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Computation of mu, alpha, beta and gamma
generate_fhmm;
states = get_all_states(M,K);
mu = zeros(K^M,D);
for i=1:K^M
    for m=1:M
        mu(i,:) = mu(i,:) + W(:,(m-1)*K+states(i,m))';
    end
end
logAlpha = alphaRecursion(Y,Pi,P,W,C);
logBeta = betaRecursion(Y,Pi,P,W,C);
gamma = Gamma(logAlpha,logBeta);
[~,temp] = max(gamma,[],2);
figure(1); hold on;
for i = 1 : K^M
    plot(Y(1,temp==i),Y(2,temp==i),'.','MarkerSize',12)
end
plot(mu(:,1),mu(:,2),'kx','MarkerSize',15,'LineWidth',3);
title 'Most likely states according to the marginal probability'
hold off;

% Exact inference
maxIter = 100;
epsilon = 0.0001;
[W,C,P,Pi,LL] = em_fhmm(Y,K,M,maxIter,epsilon);
figure(2);
plot(1:length(LL),LL,'Linewidth',2);
xlabel('Number of iteration','FontSize',18,'FontWeight','Bold');
ylabel('Log-likelihood','FontSize',18,'FontWeight','Bold');
title('Log-likelihood as a function of iteration','FontSize',18,'FontWeight','Bold');
