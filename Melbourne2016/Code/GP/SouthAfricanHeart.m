
% Reading the data
load SouthAfricanHeartData.mat

% Plotting the data
figure
plot(X(y<0,5), X(y<0,4), 'b+', 'MarkerSize', 8); hold on
plot(X(y>0,5), X(y>0,4), 'ro', 'MarkerSize', 8);
xlabel('Age','fontsize',12)
ylabel('Obesity','fontsize',12)
box off
print SAheartDataAgeObesity -depsc2


%% Setting up a grid of test inputs
[t1, t2] = meshgrid(15:1:65,10:1:50);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs

%% The prior
meanfunc = @meanZero;
covfunc = @covSEard;   hyp.cov = log([1 1 1]);  % hyp = [ log(ell_1) log(ell_2) ... log(ell_D) log(sf)]
%meanfunc = @meanConst; hyp.mean = 0;
%covfunc = @covSEard;   hyp.cov = log([1 1 1]);  % hyp = [ mu log(ell_1) log(ell_2) ... log(ell_D) ]
likfunc = @likLogistic;

%% Optimize on the hyperparameters
x = X(:,[5 4]); % Use only Age and Obesity
nTrain = length(y);
x = (x - repmat(mean(x),nTrain,1))./repmat(std(x),nTrain,1);

hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);

[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
figure
set(gca, 'FontSize', 12)
plot(X(y<0,5), X(y<0,4), 'b+', 'MarkerSize', 8); hold on
plot(X(y>0,5), X(y>0,4), 'ro', 'MarkerSize', 8);
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
box off
xlabel('Age','fontsize',12)
ylabel('Obesity','fontsize',12)
title('Laplace')
print petalMinHypEP -depsc2
