function [k, allk] = kmeans(X, K, epsilon)

    % Computes the K-means for a fixed number of centers K

    % Dimensions :
    % D : Dimension of observed points
    % N : Number of points
    % K : Number of centers

    % Input :
    % X : (D*N) : Observed points
    % K : Int   : Number of centers

    % Optionnal input :
    % epsilon : Double : Precision wanted for the k centers (norm(last_k - k) < epsilon)

    % Output :
    % k    : (D*K)     : Centers found
    % allk : ((D*K)*?) : All centers found for each iteration
    
    D = size(X,1);
    N = size(X,2);
    k = X(:,randperm(N)(1:K));

    % Compute a reasonable epsilon if not defined yet
    if (nargin < 3)
        x = X(:,randperm(N)(1:min(N,5)));
        avg_dist = sqrt(mean(mean(sqdist(x,x))));
        epsilon = avg_dist / 10000;
    end

    % Initialisations
    lastk = k*100;
    allk = reshape(k,D*K,1);
    stooop = 1;

    % K-means iterations
    while ((norm(lastk-k)>epsilon) && (stooop < 1000))

        dd = sqdist(k,X);
        [~,imin] = min(dd,[],1);
        lastk = k;
        for i=1:K
            k(:,i) = mean(X(:,imin==i),2);
        end
        allk = [allk, reshape(k,D*K,1)];
        stooop++;
    end

    
    if (stooop >= 1000)
        printf("Warning : Maximum iteration reached.\n");
    end
    
end

