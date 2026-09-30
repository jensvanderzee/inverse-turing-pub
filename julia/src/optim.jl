"""
    AdamState(n; lr, beta1 = 0.9, beta2 = 0.95, eps = 1e-8)

Adam optimiser for `n` parameters, written to match `torch.optim.Adam` term for
term (bias-corrected first and second moments, `eps` added *outside* the square
root). The defaults are the ones the Python scripts use — note `beta2 = 0.95`
rather than PyTorch's own default of `0.999`.

Kept in-package rather than pulled from Optimisers.jl so that the update rule is
visible next to the code it has to reproduce.
"""
mutable struct AdamState
    lr::Float64
    const beta1::Float64
    const beta2::Float64
    const eps::Float64
    const m::Vector{Float64}
    const v::Vector{Float64}
    t::Int
end

function AdamState(n::Integer; lr::Real, beta1::Real = 0.9, beta2::Real = 0.95, eps::Real = 1e-8)
    AdamState(Float64(lr), Float64(beta1), Float64(beta2), Float64(eps),
              zeros(Float64, n), zeros(Float64, n), 0)
end

"""
    adam_step!(opt, θ, g) -> θ

One in-place Adam update of `θ` given gradient `g`.
"""
function adam_step!(opt::AdamState, θ::AbstractVector, g::AbstractVector)
    length(θ) == length(g) == length(opt.m) ||
        throw(DimensionMismatch("parameter and gradient lengths must match the optimiser state"))
    opt.t += 1
    b1, b2 = opt.beta1, opt.beta2
    bc1 = 1 - b1^opt.t
    bc2 = 1 - b2^opt.t
    step_size = opt.lr / bc1
    sqrt_bc2 = sqrt(bc2)
    @inbounds for i in eachindex(θ)
        opt.m[i] = b1 * opt.m[i] + (1 - b1) * g[i]
        opt.v[i] = b2 * opt.v[i] + (1 - b2) * g[i]^2
        denom = sqrt(opt.v[i]) / sqrt_bc2 + opt.eps
        θ[i] -= step_size * opt.m[i] / denom
    end
    return θ
end

"""
    clip_global_norm!(g, max_norm) -> Float64

Scale `g` in place so its 2-norm does not exceed `max_norm`, and return the norm
*before* clipping. Matches `torch.nn.utils.clip_grad_norm_`, including the `1e-6`
guard in the denominator.
"""
function clip_global_norm!(g::AbstractVector, max_norm::Real)
    total = sqrt(sum(abs2, g))
    coef = max_norm / (total + 1e-6)
    coef < 1 && (g .*= coef)
    return total
end

"""
    decay_lr!(opt, gamma)

Multiplicative per-epoch learning-rate decay, equivalent to
`StepLR(optimizer, step_size=1, gamma=gamma)`.
"""
decay_lr!(opt::AdamState, gamma::Real) = (opt.lr *= gamma; opt.lr)
