data {
  int<lower=1> D;
  int<lower=0> Ntrain;
  int<lower=1> L;
  int<lower=0> Ntest;
  int<lower=0,upper=1> y[Ntrain];
  int<lower=1,upper=L> lltrain[Ntrain];
  int<lower=1,upper=L> lltest[Ntest];
  row_vector[D] xtrain[Ntrain];
  row_vector[D] xtest[Ntest];
}

parameters {
  vector[L] beta_raw;
  vector[D] beta_bar;
  vector<lower=0>[D] beta_sigma;
  real alpha_raw[L];
  real alpha_bar;
  real<lower=0> alpha_sigma;
}

transformed parameters{
  vector[Ntrain] x_beta_lltrain;
  real alpha[L];
  vector[D] beta[L];
  
  for (l in 1:L){
    alpha[l] = alpha_bar + alpha_raw[l] * alpha_sigma;
    beta[l] = beta_bar + beta_raw[l] * beta_sigma;
  }
  
  for (n in 1:Ntrain){
    x_beta_lltrain[n] = alpha[lltrain[n]] + xtrain[n] * beta[lltrain[n]];
  }
}

model {
  alpha_bar ~ student_t(7,0,2.5);
  alpha_raw ~ student_t(7,0,2.5);
  alpha_sigma ~ cauchy(0,10);
  
  beta_bar ~ student_t(7,0,2.5);
  beta_raw ~ student_t(7,0,2.5);
  beta_sigma ~ cauchy(0,2.5);
  
  y ~ bernoulli_logit(x_beta_lltrain);
}

generated quantities{
  vector[Ntest] pred_Survival;
  vector[Ntrain] pred_Survival_train;
  real log_lik[Ntrain];
  
  for (n in 1:Ntest)
    pred_Survival[n] = bernoulli_logit_rng(alpha[lltest[n]] + xtest[n] * beta[lltest[n]]); 
  
  for(i in 1:Ntrain){
    pred_Survival_train[i] = bernoulli_logit_rng(alpha[lltrain[i]] + xtrain[i] * beta[lltrain[i]]); 
    log_lik[i] = bernoulli_logit_lpmf(y[i]|x_beta_lltrain);
  }
}
