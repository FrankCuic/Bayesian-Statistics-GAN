library(rstan)
library(ggplot2)
library(dplyr)
library(readr)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data <- read_csv("STOCKDATA.csv")

data <- data %>%
  mutate(
    log_return = log(Close / lag(Close)),
    Volume_lag = lag(Volume)
  ) %>%
  na.omit()

stan_data <- list(
  N = nrow(data),
  R = data$log_return,
  L = 1,  # AR(1) model
  I = data$Volume_lag
)

stan_code <- "
data {
  int<lower=0> N;             // Number of data points
  vector[N] R;                // Log-returns
  int<lower=1> L;             // AR lag order
  vector[N] I;                // External predictor, e.g., volume
}

parameters {
  real alpha;                 // Intercept for mean equation
  real phi;                   // AR coefficient
  real<lower=0> sigma;        // Standard deviation of log-returns
  real beta;                  // Coefficient for the external predictor
}

model {
  vector[N-L] mu;

  // Priors
  alpha ~ normal(0, 10);
  phi ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  beta ~ normal(0, 1);

  // Likelihood
  for (t in (L+1):N) {
    mu[t-L] = alpha + phi * R[t-L] + beta * I[t-L];
  }

  R[(L+1):N] ~ normal(mu, sigma);
}
"

stan_model <- stan_model(model_code = stan_code)

fit <- sampling(stan_model, data = stan_data, iter = 2000, chains = 4)

print(fit)

posterior_samples <- extract(fit)

df <- as.data.frame(posterior_samples)
ggplot(df, aes(x = alpha)) + 
  geom_histogram(bins = 50, fill = "dodgerblue", color = "black") +
  labs(title = "Posterior Distribution of Alpha", x = "Alpha", y = "Density")
