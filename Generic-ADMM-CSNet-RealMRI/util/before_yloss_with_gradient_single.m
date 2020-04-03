function [ loss, ps, xx ] = before_yloss_with_gradient_single( train ,label , net )
%--------------------Hyper-Parameters---------------------
config;
LL = nnconfig.LinearLabel;

Modee = nnconfig.Modee;
gp = nnconfig.EnableGPU;
gp = 0;

relax = 0;
adapt = 0; 
N = numel(net.layers);
res = struct(...
            'x',cell(1,N+1),...
            'dzdx',cell(1,N+1),...
            'dzdw',cell(1,N+1));

res(1).x = train;   % train = \Phi^T*y, where ||\Phi*x - y||
iter = 1;
xold = res(1).x; % Ax, instead of Ax^tilda, for computing adapt and relax
gamma = 1; % TODO different init of gamma relax variable?
tau   = 1; % adapt
freq  = 2;
orthval = 0.2;
gamma0  = 1.5;
minval  = 1.e-20;
gmh     = 1.9;
gmg     = 1.1;
verbose = 0;

for j = 1:N
    l = net.layers{j};
    switch l.type
        case {'Reorg','Remid','Refinal'}
             fprintf('\n%s  %f\n', l.type, l.weights{1});
        otherwise     
                fprintf('%s\n', l.type);
    end
end
% The forward propagation
for i = 1 : N
    l = net.layers{i};
    switch l.type
       case {'Reorg','Remid','Refinal'}
            %% ADAPT RHO
       case 'ADD' %% ADD adds x + ya
	    %% RELAX FOR Y SOLVER TODO understand conv
	    %% TODO in first iter, they enter conv without ADD   
    end
    switch  l.type
       case 'Reorg'
	    if adapt
               res(i+1).x = xorg(res(i).x , tau);
%              res(i+1).x = xorg(res(i).x , l.weights{1}); %TODO should I use initial tau??
	    else 
              res(i+1).x = xorg(res(i).x , l.weights{1});
	    end
	    if relax
              xold       = res(i+1).x;
  	      res(i+1).x = gamma*res(i+1); %(1-gamma)*z_0, but z_0 is zero. (other option is to init zero, but its complex matrix, so maybe not
	    end
       case 'MIN'
            res(i+1).x = Minus(res(i-3).x , res(i).x) ; 
	    % Incorrect formulation. TODO
	    % should be 
       case 'Multi_org'
	    if adapt   
              res(i+1).x = betaorg(res(i-4).x , res(i).x , tau);
            else
              res(i+1).x = betaorg(res(i-4).x , res(i).x , l.weights{1});
            end
            % TODO multiply res(i-4) by mu2, add mu1*zold
            if adapt && relax
              yold  = res(i+1).x;
	      yhat0 = tau*(xold);% y0 = null,z0 = null, b=null
	      zold  = res(i).x;
	      xold0 = xold;
            end
	    iter = iter + 1;
       case 'Multi_mid'
            if adapt
              res(i+1).x = betamid(res(i-6).x , res(i-5).x , res(i).x , tau);
            else
              res(i+1).x = betamid(res(i-6).x , res(i-5).x , res(i).x ,l.weights{1});
            end
            % TODO multiply res(i-4) by mu2, add mu1*zold
            % ADAPT AND RELAX UPDATE
	    if adapt && relax && mod(iter,freq)==0
	      yhat = res(i+1).x + tau*(xold-res(i).x);
	      [tau,gamma] = aradmm_estimate(iter,tau,gamma,xold,xold0,yhat,yhat0,res(i).x,zold,orthval,verbose, minval, gmh, gmg, gamma0); 
	      yold  = res(i+1).x;
	      yhat0 = yhat;
	      zold  = res(i).x;
	      xold0 = xold;
	    end
	    iter = iter + 1;
       case 'Multi_final'
	    if adapt
              res(i+1).x = betafinal(res(i-6).x , res(i-5).x , res(i).x, tau);          
            else
              res(i+1).x = betafinal(res(i-6).x , res(i-5).x , res(i).x, l.weights{1});          
            end
            % TODO multiply res(i-4) by mu2, add mu1*zold
	    if adapt && relax && mod(iter,freq)==0
	      yhat = res(i+1).x + tau*(xold-res(i).x);
	      [tau,gamma] = aradmm_estimate(iter,tau,gamma,xold,xold0,yhat,yhat0,res(i).x,zold,orthval,verbose, minval, gmh, gmg, gamma0); 
	      yold  = res(i+1).x;
	      yhat0 = yhat;
	      zold  = res(i).x;
	      xold0 = xold;
	    end
	    iter = iter + 1;
       case 'ADD'
            res(i+1).x = Add (res(i).x, res(i-1).x) ;
       case 'Remid'
	    if adapt
              res(i+1).x = xmid (res(i-1).x ,res(i).x,train, l.weights{1});
            else
              res(i+1).x = xmid (res(i-1).x ,res(i).x,train, l.weights{1});
            end
	    if relax
  	      res(i+1).x = gamma*res(i+1).x + (1-gamma)*(res(i-2).x);
	    end
       case 'Refinal'
	    if adapt   
              res(i+1).x = xfinal (res(i-1).x ,res(i).x, train, l.weights{1});
            else
              res(i+1).x = xfinal (res(i-1).x ,res(i).x, train, l.weights{1});
            end
	    if relax
  	      res(i+1).x = gamma*res(i+1).x + (1-gamma)*(res(i-2).x);
	    end
       case 'c_conv'
            w1 = l.weights{1} ;
            w2 = l.weights{2} ; 
            if gp
                w1 = gpuArray(w1) ;
                w2 = gpuArray(w2) ;
            end
            res(i+1).x = comconv(res(i).x, w1, w2) ; 
       case 'conv'
            w1 = l.weights{1} ;
            w2 = l.weights{2} ; 
            if gp
                w1 = gpuArray(w1) ;
                w2 = gpuArray(w2) ;
            end
         
            res(i+1).x = vl_nnconv(res(i).x, w1, w2, ...
                                  'pad', l.pad, ...
                                  'stride', l.stride, ...
                                  'dilate', 1 ) ;
       case 'Non_linear'
            r = l.weights{1} ; 
            if gp
               r = gpuArray(r);     
               LL = gpuArray(LL);   
            end        
            res(i+1).x = nonlinear( LL , res(i).x , r);
       case 'relu'
             if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
             res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
       case 'bnorm'
             if Modee     %%%test mode
               res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                                'moments', l.weights{3}, ...
                                'epsilon', l.epsilon) ;
             else
               res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                                'epsilon', l.epsilon ) ;
              end

      case 'rLoss'
      res(i+1).x = rnnloss(res(i).x, label) ;

       otherwise
            error('No such layers type.');
    end

%     %% ADAPT. ARADMM
%        if iter == 1 %record at first iteration
%            l0 = l;
%            l_hat0 = l1 + tau*(b-Au-Bv1);
%            Bv0 = Bv; 
%            Au0 = Au; 
%        elseif mod(iter,freq)==0 && iter>siter && iter < eiter   %adaptive stepsize
%            %l_hat
%            l_hat = l1 + tau*(b-Au-Bv1);
% 
% %          [tau, gamma] = aradmm_estimate(iter, tau, gamma, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, orthval, minval, verbose, gmh, gmg, gamma0);
% 
%            % record for next estimation
%            l0 = l;
%            l_hat0 = l_hat;
%            Bv0 = Bv; 
%            Au0 = Au; 
%        end %frequency if, AADMM
% 
%     switch l.type
%       case {'Multi_org','Multi_mid','Multi_final'}
%         iter = iter +1;
    
end


   loss = res(end).x;
   
    loss = double(loss);
    xx = res(end-1).x;
    xx = gather(xx);
    label = gather(label);
    ps = psnr(abs(xx), abs(label));
    

end


function tau_h = curv_adaptive_BB(al_h, de_h)
%adapive BB, reference: FASTA paper of Tom
tmph = de_h/al_h; %correlation
if tmph > .5
    tau_h = de_h;
else
    tau_h = al_h - 0.5*de_h;
end
end

function [tau, gamma] = aradmm_estimate(iter, tau, gamma, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, orthval, minval, verbose, gmh, gmg, gamma0)
%inner product
tmp = real(conj(Au-Au0).*(l_hat-l_hat0));
ul_hat = sum(tmp(:));
tmp = real(conj(Bv-Bv0).*(l-l0));
vl = sum(tmp(:));

%norm of lambda, lambda_hat
tmp = l_hat-l_hat0;
dl_hat = norm(tmp(:));
tmp = l-l0;
dl = norm(tmp(:));

%norm of gradient change
tmp = Au-Au0;
du = norm(tmp(:));
tmp = Bv-Bv0;
dv = norm(tmp(:));


%flag to indicate whether the curvature can be estimated
hflag = false;
gflag = false;

%estimate curvature, only if it can be estimated
%use correlation/othogonal to test whether can be estimated
%use the previous stepsize when curvature cannot be estimated
if ul_hat > orthval*du*dl_hat + minval
    hflag = true;
    al_h = dl_hat^2/ul_hat;
    de_h = ul_hat/du^2;
    bb_h = curv_adaptive_BB(al_h, de_h);
end
if vl > orthval*dv*dl + minval
    gflag = true;
    al_g = dl^2/vl;
    de_g = vl/dv^2;
    bb_g = curv_adaptive_BB(al_g, de_g);
end

if hflag && gflag
    ss_h = sqrt(bb_h);
    ss_g = sqrt(bb_g);
    gamma = min(1 + 2/(ss_h/ss_g+ss_g/ss_h), gamma0);
    tau = ss_h*ss_g;
elseif hflag
    gamma = gmh; %1.9;
    tau = bb_h;
elseif gflag
    gamma = gmg; %1.1;
    tau = bb_g;
else
    gamma = gamma0; %1.5;
    %tau = tau;
end

if verbose == 3
    if ul_hat < 0 || vl < 0
        fprintf('(%d) <u, l>=%f, <v, l>=%f\n', iter, ul_hat, vl);
    end
    if hflag
        fprintf('(%d) corr_h=%f,  al_h=%f,  estimated tau=%f,  gamma=%f \n', iter,...
            ul_hat/du/dl_hat, al_h, tau, gamma);
    end
    if gflag
        fprintf('(%d) corr_g=%f,  al_g=%f,  tau=%f gamma=%f\n', iter,...
            vl/dv/dl, al_g, tau, gamma);
    end
    if ~hflag && ~gflag
        fprintf('(%d) no curvature, corr_h=%f, corr_g=%f, tau=%f,  gamma=%f\n', iter,...
            ul_hat/du/dl_hat, vl/dv/dl, tau, gamma);
    end
end
end

    
            










