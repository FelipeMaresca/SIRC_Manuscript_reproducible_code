# ---------------------------------------------------------------------------
# compute_fig6_data.jl -- computes the data of Fig. 6 (heat maps) and writes it to data_fig_6/*.csv   (PARALLEL version)
#
# The Python notebook reads these files (folder `data_fig_6/`, next to the notebook) instead of recomputing them
# with scipy. Run this script from the folder that contains the notebook, using several threads:
#
#     julia --project=. -t auto compute_fig6_data.jl          # all available cores
#     julia --project=. -t 32 compute_fig6_data.jl            # or an explicit number of threads
#
# (a single thread also works, it just takes longer). Requires the packages DifferentialEquations and
# DelimitedFiles (the latter is part of Julia).
#
# Parallelisation: every integration is independent, so each matrix is filled by `nthreads()` workers that
# pull the next (row, column) pair from a shared atomic counter (dynamic load balancing: cheap and expensive
# integrations are mixed evenly among the threads). The results do not depend on the number of threads.
#
# Resuming: a panel whose csv already exists in `data_fig_6/` is skipped (delete the file, or set
# RESUME = false below, to recompute it). Useful if the job is interrupted on the server.
# ---------------------------------------------------------------------------
using DifferentialEquations, DelimitedFiles

const RESUME = true

# ---- SIRC model (same equations and parameters as in the manuscript) --------------------------------------
const BETA, GAMMA, OMEGA, I0 = 0.2, 0.1, 0.025, 1e-6

# Solver tolerances (same as the Julia notebook for reltol). rho_i can fall to 1e-30 or less between outbreaks,
# so its absolute tolerance is tiny: this keeps rho_i positive (as in the Python notebook, where
# atol = [1e-10, 1e-10, 1e-60, 1e-10]); a negative rho_i would make later outbreaks blow up or vanish.
const RELTOL = 1e-8
const ABSTOL = [1e-10, 1e-10, 1e-60, 1e-10]

# State u = [rho_c, rho_s, rho_i, rho_r]; p = (is_alpha, a).
# is_alpha = true : confinement rate Gamma = a        (spontaneous confinement, a = alpha)
# is_alpha = false: confinement rate Gamma = a*rho_i  (prevalence-driven confinement, a = eta)
function sirc!(du, u, p, t)
    is_alpha, a = p
    rho_c, rho_s, rho_i, rho_r = u
    Gamma = is_alpha ? a : a * rho_i
    du[1] = Gamma * rho_s - OMEGA * rho_c
    du[2] = OMEGA * rho_c - Gamma * rho_s - BETA * rho_s * rho_i
    du[3] = BETA * rho_s * rho_i - GAMMA * rho_i
    du[4] = GAMMA * rho_i
end

# Integrate from t0 to t1 with a constant value `a` of the control; returns the final state.
function advance(u, t0, t1, is_alpha, a)
    prob = ODEProblem(sirc!, u, (t0, t1), (is_alpha, a))
    sol = solve(prob, Tsit5(); reltol = RELTOL, abstol = ABSTOL, save_everystep = false)
    return sol.u[end]
end

# Final size rho_r(t_end) for a campaign of strength `a` that starts at t_ini and lasts tau,
# but cannot go past the deadline T (the control is zero outside [t_ini, min(t_ini + tau, T)]).
function campaign_final_size(is_alpha, a, t_ini, tau; T = 250.0, t_end = 3000.0)
    u = [0.0, 1.0 - I0, I0, 0.0]
    t_off = min(t_ini + tau, T)
    if t_off > t_ini && t_ini < t_end
        t_ini > 0 && (u = advance(u, 0.0, t_ini, is_alpha, 0.0))   # before the campaign
        u = advance(u, t_ini, t_off, is_alpha, a)                  # campaign
        u = advance(u, t_off, t_end, is_alpha, 0.0)                # after the campaign
    else
        u = advance(u, 0.0, t_end, is_alpha, 0.0)                  # no campaign inside the window
    end
    return u[4]
end

# ---- parallel helper -------------------------------------------------------------------------------------
# R[i, j] = f(i, j) for every entry of the preallocated matrix R, computed by all the threads.
function parallel_fill!(R::AbstractMatrix, f)
    n, nr = length(R), size(R, 1)
    next = Threads.Atomic{Int}(1)                       # next linear index (column-major) to be computed
    Threads.@threads for _ in 1:Threads.nthreads()      # one worker per thread
        while true
            idx = Threads.atomic_add!(next, 1)
            idx > n && break
            i = (idx - 1) % nr + 1
            j = (idx - 1) ÷ nr + 1
            R[i, j] = f(i, j)
        end
    end
    return R
end

println("Threads.nthreads() = ", Threads.nthreads(),
        Threads.nthreads() == 1 ? "   (run with `julia --project=. -t auto compute_fig6_data.jl` to parallelise)" : "")
mkpath("data_fig_6")

# ---- Figure 6: one matrix per panel; rows = tau (1..250), columns = t_ini (0..250) -----------------------
const TAUS   = collect(1.0:250.0)
const TINIS  = collect(0.0:250.0)
panels = [(true, 0.005), (true, 0.01), (true, 0.5), (false, 0.25), (false, 0.35), (false, 1.0)]
for (k, (is_alpha, a)) in enumerate(panels)
    file = "data_fig_6/fig6_step1_panel$(k).csv"
    if RESUME && isfile(file)
        println("Figure 6, panel $k: $file already exists, skipped")
        continue
    end
    t0 = time()
    R = Matrix{Float64}(undef, length(TAUS), length(TINIS))                     # shape 250 x 251
    parallel_fill!(R, (i, j) -> campaign_final_size(is_alpha, a, TINIS[j], TAUS[i]))
    writedlm(file, R, ',')
    println("Figure 6, panel $k done  ($(round(time() - t0; digits = 1)) s)")
end

println("All done.")
