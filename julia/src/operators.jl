"""
    laplacian!(out, u) -> out

Five-point discrete Laplacian with replicate (zero-flux / Neumann) boundaries.

This reproduces the PyTorch operator used throughout the original code:

    nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode="replicate", bias=False)

with the fixed kernel `[0 1 0; 1 -4 1; 0 1 0]`. Out-of-domain neighbours take the
value of the nearest edge cell, so a cell on the boundary effectively sees itself
as its own outside neighbour.

Grid spacing is folded into the diffusion coefficients (the kernel carries no
`1/h^2`), exactly as in the Python version — the fitted coefficients are therefore
in units of "cells squared per time unit", not metres squared.

Two implementations are provided: a cache-friendly loop for contiguous CPU arrays,
and a shift-and-accumulate broadcast version for every other array type. The latter
is free of scalar indexing, so a GPU array (`CuArray`, `ROCArray`, ...) works
without any further code.
"""
function laplacian! end

"""
Contiguous CPU matrices: a plain `Array`, or a slice like `view(u, :, :, k)` of one.

The slice case is what the ODE backend produces when it splits a packed `H×W×3`
state into fields, and it must reach the fast loop rather than the generic
fallback — dispatching on `Array` alone would silently cost several times the
runtime there.

Deliberately *not* `StridedMatrix`: `CuArray` is a `DenseArray` and would match it,
sending GPU data into a scalar-indexed loop.
"""
const ContiguousCPUMatrix = Union{Array{<:Any,2},
                                  SubArray{<:Any,2,<:Array,<:Any,true}}

function laplacian!(out::ContiguousCPUMatrix, u::ContiguousCPUMatrix)
    axes(out) == axes(u) || throw(DimensionMismatch("out and u must have equal axes"))
    H, W = size(u)
    @inbounds for j in 1:W
        jm = j == 1 ? 1 : j - 1
        jp = j == W ? W : j + 1
        # Top row: the upward neighbour replicates the row itself.
        out[1, j] = u[1, j] + u[min(2, H), j] + u[1, jm] + u[1, jp] - 4 * u[1, j]
        @simd for i in 2:(H - 1)
            out[i, j] = u[i - 1, j] + u[i + 1, j] + u[i, jm] + u[i, jp] - 4 * u[i, j]
        end
        if H > 1
            # Bottom row: the downward neighbour replicates the row itself.
            out[H, j] = u[H - 1, j] + u[H, j] + u[H, jm] + u[H, jp] - 4 * u[H, j]
        end
    end
    return out
end

function laplacian!(out::AbstractMatrix, u::AbstractMatrix)
    axes(out) == axes(u) || throw(DimensionMismatch("out and u must have equal axes"))
    H, W = size(u)
    @. out = -4 * u
    # Vertical neighbours, with the first/last row replicating themselves.
    @views @. out[2:H, :] += u[1:(H - 1), :]
    @views @. out[1, :] += u[1, :]
    @views @. out[1:(H - 1), :] += u[2:H, :]
    @views @. out[H, :] += u[H, :]
    # Horizontal neighbours, likewise for the first/last column.
    @views @. out[:, 2:W] += u[:, 1:(W - 1)]
    @views @. out[:, 1] += u[:, 1]
    @views @. out[:, 1:(W - 1)] += u[:, 2:W]
    @views @. out[:, W] += u[:, W]
    return out
end

"""
    laplacian(u)

Allocating form of [`laplacian!`](@ref); convenient for tests and one-off checks.
"""
laplacian(u::AbstractMatrix) = laplacian!(similar(u), u)
