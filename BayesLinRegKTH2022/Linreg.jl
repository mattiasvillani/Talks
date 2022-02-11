using Distributions, LinearAlgebra, PDMats

"""
  Maximum likelihood estimates (unbiased for σ^2) in linear regression
"""
function LinRegMLE(y::Vector, X, addintercept=false)
  n,p = size(X)
  if addintercept 
      X = [ones(n,1) X]
  end
  βhat =	X\y
  s2 = (y-X*βhat)'*(y-X*βhat)/(n-p)
  βcov = s2*inv(X'*X)
  βse = .√diag(βcov)
  tratio = βhat./βse
  return βhat, βcov, βse, tratio
end


# Def Inv-χ²(ν,τ²) using InverseGamma in Distributions.jl
ScaledInverseChiSq(ν,τ²) = InverseGamma(ν/2,ν*τ²/2)

"""
    BayesLinReg(y, X, μ₀, Ω₀, ν₀, σ²₀, nSim)
Bayesian Linear regression with a conjugate prior.
Model: 
```math
\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\varepsilon}
\\boldsymbol{\\varepsilon}\\sim\\mathrm{N}(0,\\sigma^2)
```
Prior:
```math
\\boldsymbol{\\beta} | \\sigma^2 \\sim \\mathrm{N}(\\mu_0,\\sigma^2\\Omega_0^{-1})
\\sigma^2 \\sim \\mathrm{Inv-}\\chi^2(\\nu_0,\\sigma_0^2)
```

INPUTS:
y       n×1 vector with response data observations
X       n×p matrix with covariates. No intercept is added automatically.
μ₀      p×1 vector with prior mean for β
Ω₀      p×p prior precision matrix for β
ν₀      prior degrees of freedom for σ² 
σ²₀     prior location for σ² 
nSim    Number of draws from the joint posterior p(β,σ²|y,X)

OUTPUTS:
μₙ      p×1 vector with posterior mean for β
Ωₙ      p×p posterior precision matrix for β
νₙ      posterior degrees of freedom for σ²
σ²ₙ     posterior location for σ² 
βsim    nSim×p matrix with posterior draws for β
σ²sim   nSim×1 vector with posterior draws for σ²

```julia
μₙ, Ωₙ, νₙ, σ²ₙ, βsim, σ²sim = BayesLinReg(y, X, μ₀, Ω₀, ν₀, σ²₀, nSim)
```
External links
* [Bayesian Learning Book](https://github.com/mattiasvillani/BayesianLearningBook)
"""

function BayesLinReg(y::Vector, X, μ₀, Ω₀, ν₀, σ²₀, nSim)
    
    # Compute posterior hyperparameters
    n = length(y) 
    p = size(X,2)
    XX = X'*X
    βhat = X \ y
    Ωₙ = Symmetric(XX + Ω₀)
    μₙ = Ωₙ\(XX*βhat + Ω₀*μ₀)
    νₙ = ν₀ + n
    σ²ₙ = (ν₀*σ²₀ + (y-X*βhat)'*(y-X*βhat) + 
          (μₙ-βhat)'*XX*(μₙ-βhat) + 
          (μₙ-μ₀)'*Ω₀*(μₙ-μ₀) 
          )/νₙ

    # Sampling from posterior
    invΩₙ = inv(Ωₙ)
    σ²sim = zeros(nSim)
    βsim = zeros(nSim,p)
    for i ∈ 1:nSim
        # Simulate from p(σ²|y,X)
        σ² = rand(ScaledInverseChiSq(νₙ,σ²ₙ))
        σ²sim[i] = σ²
 
        # Simulate from p(β|σ²,y,X)
        β = rand(MvNormal(μₙ,σ²*invΩₙ))
        βsim[i,:] = β'

    end
  return μₙ, Ωₙ, νₙ, σ²ₙ, βsim, σ²sim
end

# n = 100
# X = randn(n,2)
# X = [ones(n) X]
# βtrue = [1 2 3]'
# σ²true = 0.5
# y = (X*βtrue)[:] +  √σ²true*randn(n)
# μ₀ = zeros(3)
# τ = 10
# Ω₀ = inv(τ^2*I(3))
# ν₀ = 3
# σ²₀ = 1
# nSim = 10000
# μₙ, Ωₙ, νₙ, σ²ₙ, βsim, σ²sim = BayesLinReg(y, X, μ₀, Ω₀, ν₀, σ²₀, nSim);
# mean(σ²sim)
# mean(βsim, dims = 1)
# p1 = histogram(.√σ²sim, bins = 50, title = L"\sigma", lw = 0, alpha = 0.8)
# p2 = histogram(βsim[:,2], bins = 50, title = L"\beta_2", lw = 0, alpha = 0.8)
# plot(p1,p2, layout = (1,2))

# Simulate from prior
function BayesLinRegPrior(μ₀, Ω₀, ν₀, σ²₀, nSim)
  p = size(Ω₀,2)
  Ω₀ = Symmetric(Ω₀)
  invΩ₀ = inv(Ω₀)
  σ²sim = zeros(nSim)
  βsim = zeros(nSim,p)
  for i ∈ 1:nSim
      # Simulate from p(σ²)
      σ² = rand(ScaledInverseChiSq(ν₀,σ²₀))
      σ²sim[i] = σ²

      # Simulate from p(β|σ²)
      β = rand(MvNormal(μ₀,σ²*invΩ₀))
      βsim[i,:] = β'

  end
  return βsim, σ²sim
end


# Marginal likelihood - σ² known
function LogMargLikeGaussReg(y::Vector, X, μ₀, Ω₀, σ²)
  # Variance σ² assumed known
    return logpdf(MvNormal(X*μ₀,σ²*(I(n) + X*inv(Ω₀)*X'), y))
end

# Marginal likelihood - σ² unknown
function LogMargLikeGaussReg(y, X, μ₀, Ω₀, ν₀, σ²₀)
  n = length(y)
  Σ = PDMat(Symmetric(σ²₀*(I(n) + X*inv(Ω₀)*X')))
  return logpdf(MvTDist(ν₀, X*μ₀, Σ), y)
end