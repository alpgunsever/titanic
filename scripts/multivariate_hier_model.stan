// This code chunk is modified based on Stan User Guide's section 1.13 Multivariate Priors for Hierarchical Models 
data {
  int<lower=0> Ntrain;         // num individuals training set
  int<lower=0> Ntest;         // num individuals test set
  int<lower=1> K;              // num ind predictors
  int<lower=1> J;              // num groups
  int<lower=1,upper=J> jjtrain[Ntrain];  // group for individuals in training set
  int<lower=1,upper=J> jjtest[Ntest];  // group for individuals in test set
  matrix[Ntrain,K] xtrain;               // individual predictors in training set
  matrix[Ntest,K] xtest;               // individual predictors in test set
  int<lower=0,upper=1> y[Ntrain];
}
parameters {
  matrix[J,K] mu;     // group level mean vector for each individual coefficient
  matrix[K, J] z;
  cholesky_factor_corr[K] L_Omega;
  vector<lower=0,upper=pi()/2>[K] tau_unif;
}
transformed parameters {
  vector[Ntrain] x_beta_jj;
  matrix[J, K] beta;
  vector<lower=0>[K] tau;     // prior scale
  
  for (k in 1:K){
    tau[k] = 2.5 * tan(tau_unif[k]);
  } 
  beta = mu + (diag_pre_multiply(tau,L_Omega) * z)';
  x_beta_jj = rows_dot_product(beta[jjtrain] , xtrain);
}
model {
  to_vector(z) ~ std_normal();
  L_Omega ~ lkj_corr_cholesky(0.1);
  to_vector(mu) ~ student_t(7,0,2.5);
  
  y ~ bernoulli_logit(x_beta_jj);
}
generated quantities{
  vector[Ntest] pred_Survival;
  vector[Ntrain] pred_Survival_train;
  real log_lik[Ntrain];
  vector[Ntest] x_beta_jj_test;
  
  x_beta_jj_test = rows_dot_product(beta[jjtest] , xtest);
  
  for (n in 1:Ntest)
    pred_Survival[n] = bernoulli_logit_rng(x_beta_jj_test[n]); 
  
  for(i in 1:Ntrain){
    pred_Survival_train[i] = bernoulli_logit_rng(x_beta_jj[i]); 
    log_lik[i] = bernoulli_logit_lpmf(y[i]|x_beta_jj);
  }
}
