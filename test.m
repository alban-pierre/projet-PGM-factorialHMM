%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y = [Y_1,...,Y_T]                         D  * T
% S_t = [S^{1}|...|S^{M}]'                  MK * 1
% S = [S_1,...,S_T]                         MK * T
% W = [W^{1}|...|W^{M}]                     D  * MK
% P_{k,l} = P(S_{t+1} = l | S_{t+j} = k) 
% P = [P^{1}|...|P^{M}]]'                   MK * K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Computations
generate_fhmm;
states = get_all_states(M,K);
mu = computeMu(W,states);
Ptrans = computePtrans(P,states);
gauss = computeGaussian(Y,mu,C);
logAlpha = alphaRecursion(Y,Pi,Ptrans,states,gauss);
logBeta = betaRecursion(Y,Pi,Ptrans,gauss);

% Test gamma
gamma = Gamma(logAlpha,logBeta);
[~,temp] = max(gamma,[],2);
figure(1); hold on;
for i = 1 : K^M
    plot(Y(1,temp==i),Y(2,temp==i),'.','MarkerSize',12)
end
plot(mu(:,1),mu(:,2),'kx','MarkerSize',15,'LineWidth',3);
title 'Most likely states according to the marginal probability'
hold off;

%Parameters
maxIter = 100;
epsilon = 1e-5;

% Exact inference
[W1,C1,P1,Pi1,LL1] = em_fhmm(Y,K,M,maxIter,epsilon);
figure(2);
plot(1:length(LL1),LL1,'Linewidth',2);
xlabel('Number of iteration','FontSize',18,'FontWeight','Bold');
ylabel('Log-likelihood','FontSize',18,'FontWeight','Bold');
title('Log-likelihood as a function of iteration','FontSize',18,'FontWeight','Bold');

% Gibbs sampling inference
if (false) % Does not work yet
[W2,C2,P2,Pi2,LL2] = em_gibbs(Y,K,M,maxIter,epsilon);
figure(2);
plot(2:length(LL2),LL2(2:end),'Linewidth',2);
xlabel('Number of iteration','FontSize',18,'FontWeight','Bold');
ylabel('Log-likelihood','FontSize',18,'FontWeight','Bold');
title('Log-likelihood as a function of iteration','FontSize',18,'FontWeight','Bold');
end

% Completely factorized variational inference
[W3,C3,P3,Pi3,LL3] = em_cfva(Y,K,M,maxIter,epsilon);
figure(3);
plot(2:length(LL3),LL3(2:end),'Linewidth',2);
xlabel('Number of iteration','FontSize',18,'FontWeight','Bold');
ylabel('Log-likelihood','FontSize',18,'FontWeight','Bold');
title('Log-likelihood as a function of iteration','FontSize',18,'FontWeight','Bold');
