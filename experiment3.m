% For this file the dataset "Activity Recognition from Single Chest-Mounted
% Accelerometer Data Set" from UCI Repository is needed
% The files "Data/1.csv" must exist


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


% Collect 7 types of data of size T=100, D=3
d = load('Data/1.csv');

b = 1;
T = 100;
i = d(b,5);
ndata = ones(7,1);
clear data;
for n=1:size(d,1)
	if n == b+T-1
		data{ndata(i,1), i} = d(b:n,2:4)';
		ndata(i,1) = ndata(i,1) + 1;
		b = n+1;
	end
	if d(n,5) ~= i
		b = n;
		i = d(b,5);
	end
end

ndata = ndata-1;

