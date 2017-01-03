% For this file the dataset "CalIt2 Building People
% Counts Data Set" from UCI Repository is needed
% The files "Data2/data.csv" must exist

% It fails miserably

% Variable that is 1 if we use matlab, and 0 otherwise
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;

% To be able to repeat
if (isMatlab)
    rng('default');
    rng(1);
else
    pkg load statistics;
    randn('seed',2);
    rand('seed',2);
end


d = load('Data2/data.csv');
d = reshape(d, 2, 48*15*7);



Y = d(:,1:48*10);
Y_test = d(:, 48*20+(1:48*10));


K = 2;
M = 3;
maxIter = 100;
epsilon = 1e-4;


[W0, P0, C0] = recursive_kmeans_init(Y, M, K); % sometimes it fails because too many points are exactly the same
[W,C,P,Pi,ll] = em_fhmm(Y,K,M,maxIter,epsilon,W0,P0,C0);


% 2D plot of points
alls = eye(K);
for m=2:M
    alls = reshape(repmat(alls,1,K)',K*(m-1),K^m)';
    alls = [alls, repmat(eye(K), size(alls,1)/K,1)];
end

mu = W*alls';

p = zeros(M,K);
for m=1:M
    [v,d] = eig(P((m-1)*K+1:m*K,:)');
    p(m,:) = v(:,(abs(diag(d) - 1) < 0.000001)');
end
p = p ./ sum(p,2);  
p = repmat(reshape(p',K*M,1), 1, K^M).*alls';
p = prod(reshape(p(alls'>0.5),M,K^M),1);

figure(1); hold off;
plot(Y(1,:), Y(2,:), '.b');
hold on;
for i=1:K^M
    plot(mu(1,i), mu(2,i), 'ko', 'MarkerSize',10+round(p(1,i)*200),'LineWidth',1);
end
plot(mu(1,:), mu(2,:), 'rx', 'MarkerSize',15,'LineWidth',3);




LL = loglikelihood(Y_test,W,C,P,Pi)

figure(2)
plot(1:size(ll,2), ll)
hold on;
plot(size(ll,2), LL)
