% Variable that is 1 if we use matlab, and 0 otherwise
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;

% To be able to repeat
if (isMatlab)
    rng('default');
    rng(1);
else
    pkg load statistics;
    randn('seed',8);
    rand('seed',8);
end

% Parameters
D = 2;
T = 100;
maxIter = 100;
epsilon = 1e-4;
repeat = 15;

LL0f = []; LL0testf = [];
LL1f = []; LL1testf = [];
LL2f = []; LL2testf = [];
LL3f = []; LL3testf = [];
LL4f = []; LL4testf = [];

% Comparison performance
for M = [3,5]
    for K = [2,3]
        LL0 = []; LL0test = [];
        LL1 = []; LL1test = [];
        LL2 = []; LL2test = [];
        LL3 = []; LL3test = [];
        LL4 = []; LL4test = [];
        
        for t = 1:repeat
            % Generate
            [Y,Ytest,Pi,P,W,C] = generate_fhmm(T,K,M,D);
            
            % Init
            W0 = randn(D,M*K);
            P0 = rand(M*K,K);
            %[W0, P0, C0] = recursive_kmeans_init(Y, M, K);
            P0 = P0 ./ sum(P0,2);
            
            % Exec
            [W1,C1,P1,Pi1,ll1] = em_fhmm(Y,K,M,maxIter,epsilon,W0,P0);
            [W2,C2,P2,Pi2,ll2] = em_gibbs(Y,K,M,maxIter,epsilon,W0,P0);
            [W3,C3,P3,Pi3,ll3] = em_cfva(Y,K,M,maxIter,epsilon,W0,P0);
            try     % To avoid errors
                [W4,C4,P4,Pi4,ll4] = em_sva(Y,K,M,maxIter,epsilon,W0,P0);
                LL4 = [LL4 , ll4(end)];
                LL4test = [LL4test , loglikelihood(Ytest,W4,C4,P4,Pi4)];
            catch
                fprintf('Error sva : M = %d, K = %d, rep = %d\n',M,K,t);
                continue
            end
            
            % Compute log-likelihood training and test set
            LL0 = [LL0 , loglikelihood(Y,W,C,P,Pi)];
            LL0test = [LL0test , loglikelihood(Ytest,W,C,P,Pi)];
            LL1 = [LL1 , ll1(end)];
            LL1test = [LL1test , loglikelihood(Ytest,W1,C1,P1,Pi1)];
            LL2 = [LL2 , ll2(end)];
            LL2test = [LL2test , loglikelihood(Ytest,W2,C2,P2,Pi2)];
            LL3 = [LL3 , ll3(end)];
            LL3test = [LL3test , loglikelihood(Ytest,W3,C3,P3,Pi3)];
        end 
        
        % Mean and standard deviation of log-likelihood
        LL0f = [LL0f,[mean(LL0);std(LL0)]];
        LL0testf = [LL0testf,[mean(LL0test);std(LL0test)]];
        LL1f = [LL1f,[mean(LL1);std(LL1)]];
        LL1testf = [LL1testf,[mean(LL1test);std(LL1test)]];
        LL2f = [LL2f,[mean(LL2);std(LL2)]];
        LL2testf = [LL2testf,[mean(LL2test);std(LL2test)]];
        LL3f = [LL3f,[mean(LL3);std(LL3)]];
        LL3testf = [LL3testf,[mean(LL3test);std(LL3test)]];
        LL4f = [LL4f,[mean(LL4);std(LL4)]];
        LL4testf = [LL4testf,[mean(LL4test);std(LL4test)]];
    end
end



