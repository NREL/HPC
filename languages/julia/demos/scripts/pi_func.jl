
using Random
using Statistics

using MPI

"""
    init_mpi()

Initialize the non-dynamic library parts of MPI.jl.
"""
function init_mpi()
    for f in MPI.mpi_init_hooks
        f()
    end
    return
end

function estimate_pi(comm::MPI.Comm)::Union{Float64,Nothing}
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        N = Vector{Int}([10^6])
    else
        N = Vector{Int}(undef, 1)
    end

    MPI.Bcast!(N, 0, comm)
    n = N[1]

    Random.seed!(rank)
    val = 4.0 * mean(rand(n).^2 + rand(n).^2 .< 1.0)

    rbuf = MPI.Gather(val, 0, comm)

    return rank == 0 ? mean(rbuf) : nothing
end

function cv_estimate_pi(comm::MPI.Comm, a::Float64)::Union{Tuple{Float64,Float64},Nothing}
    if (a < 1.0 || a > 2.0)
        @error("Value for a must be in [1.0,2.0] -- got a = $a")
        return nothing
    end
    
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    N = Int(10^6)
    Random.seed!(rank)
    x = rand(N)
    y = rand(N)

    z_i = x.^2 + y.^2 .<= 1.0
    z_hat = mean(z_i)
    sz_hat = var(z_i, mean=z_hat)
    w3_i = x + y .<= a
    w3_hat = mean(w3_i)
    sw3_hat = var(w3_i, mean=w3_hat)
    s_zw3_hat = cov(w3_i, z_i)
    alpha3 = -s_zw3_hat / sw3_hat
    rho3_sq = s_zw3_hat^2 / (sz_hat * sw3_hat)

    estimate = 4*(z_hat + alpha3*(w3_hat - 1 + 0.5 * (2.0 - a)^2))
    factor = 1.0 - rho3_sq

    rbuf = MPI.Gather([estimate, factor], 0, comm)

    if rank == 0
        idx = BitArray(collect(1:length(rbuf)) .% 2)
        value = mean(rbuf[idx])
        factor = mean(rbuf[.!idx])
    end

    return rank == 0 ? (value, factor) : nothing
end
