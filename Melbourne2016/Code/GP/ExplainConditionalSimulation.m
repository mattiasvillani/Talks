%% Simulating data from GP

% Setting up mean and covariance functions in the estimated GP
meanfuncEst = {@meanConst}; hypEst.mean = [0]; % Zero mean
covfuncEst = {@covSEiso}; ell = 0.2; sf = 0.5; hypEst.cov = log([ell; sf]); % Squared Exponential, isotropic. length scale is ell. signal variance is sf^2
likfuncEst = @likGauss; sn = 0.0; hypEst.lik = log(sn); % Gaussian likelihood, noise-free

% Making up some data
x = [0.5 -0.5 0.4]';
y = [0.7 0.2 0.8]';
z = linspace(-1, 1, 100)';

% First with no data (trick this with huge data variance)
hypEst.lik = log(100000);
[ymu ys2 fmu fs2] = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x(1), y(1), z);

figure('name','data and predicted curve with posterior error bands') 
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor','none');
set(gca, 'xlim',[-1.05,1.05], 'ylim',[-1.25,1.25])
box off
xlabel('x', 'fontsize',14)
ylabel('f(x)', 'fontsize',14)
title(['Length scale l = ',num2str(ell)], 'fontsize',14)
print ConditionalSim0 -depsc2

% Now with a single data point
hypEst.lik = log(0);
[ymu ys2 fmu fs2] = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x(1), y(1), z);

figure('name','data and predicted curve with posterior error bands') 
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor','none');
hold on
plot(x(1),y(1),'o')
set(gca, 'xlim',[-1.05,1.05], 'ylim',[-1.25,1.25])
box off
xlabel('x', 'fontsize',14)
ylabel('f(x)', 'fontsize',14)
title(['Length scale l = ',num2str(ell)], 'fontsize',14)
print ConditionalSim1 -depsc2

% Now with a two data points
[ymu ys2 fmu fs2] = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x(1:2), y(1:2), z);

figure('name','data and predicted curve with posterior error bands') 
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor','none');
hold on
plot(x(1),y(1),'o')
plot(x(2),y(2),'o')
set(gca, 'xlim',[-1.05,1.05], 'ylim',[-1.25,1.25])
box off
xlabel('x', 'fontsize',14)
ylabel('f(x)', 'fontsize',14)
title(['Length scale l = ',num2str(ell)], 'fontsize',14)
print ConditionalSim2 -depsc2

% Now with a three data points
[ymu ys2 fmu fs2] = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x(1:3), y(1:3), z);

figure('name','data and predicted curve with posterior error bands') 
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor','none');
hold on
plot(x(1),y(1),'o')
plot(x(2),y(2),'o')
plot(x(3),y(3),'o')
set(gca, 'xlim',[-1.05,1.05], 'ylim',[-1.25,1.25])
box off
xlabel('x', 'fontsize',14)
ylabel('f(x)', 'fontsize',14)
title(['Length scale l = ',num2str(ell)], 'fontsize',14)
print ConditionalSim3 -depsc2

