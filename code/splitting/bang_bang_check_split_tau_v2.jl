# ---------------------------------------------------------------------------------------------------------
# bang_bang_check_split_tau_v2.jl -- Is it better to split the campaign time tau into two pulses than to use a single pulse?
#
# For a total campaign time tau, compare
#   * SINGLE pulse: control on during [t_ini, t_ini + tau], with the OPTIMAL start time t_ini*;
#   * TWO pulses  : first pulse of length f*tau starting at t_ini, a gap of length `gap` with the control off,
#                   and a second pulse of length (1-f)*tau; for every pair (f, gap) the start time t_ini is
#                   also optimised, so every combination is compared at its own best t_ini.
# Everything is cut at the deadline T: the control is zero after T. A pulse that reaches T is shortened, so
# a late second pulse loses part of tau (by design: this is also what happens with a single pulse).
# The quantity compared is the final size rho_r(t_end). delta = rho_r*(two pulses) - rho_r*(single pulse);
# delta < 0 means that splitting is better. The "worst case" is the (tau, model, strength, f, gap) with the
# most negative delta: the one where the optimal two-pulse campaign differs most from the optimal single one.
#
# Cases: alpha model, alpha = 0.01, 0.06, ..., 0.96 (20 values) and eta model, 20 values of eta from 0.01 to 3;
# total campaign times tau = 20, 30, 50, 80.
#
# Usage (from the code/ folder; the script is in splitting/; results go to data/split/):
#     julia --project=. -t 30 --heap-size-hint=8G splitting/bang_bang_check_split_tau_v2.jl            # all the values of tau
#     julia --project=. -t 30 --heap-size-hint=8G splitting/bang_bang_check_split_tau_v2.jl 20 30      # only some values of tau (to run it in pieces)
#     julia --project=. -t 10 splitting/bang_bang_check_split_tau_v2.jl validate    # FAST CHECK (tau = 50, the six panels of Fig. 5; data/split_validate/)
#         compare data/split_validate/*.csv with verify_tau50.csv (independent Python implementation):
#         columns there: model, strength, f, gap, t_ini*, rho_r*. The (f = 1, gap = 0) rows are the single pulses.
#   * Do not use more than 30 threads (the script refuses to run with more).
#   * --heap-size-hint makes Julia's garbage collector keep the memory below ~8 GB (change it as needed), so
#     the server does not need swap. The memory use is small anyway: no trajectory is stored, only the
#     final state of each integration.
#   * It is safe to interrupt it and run it again: every (tau, model, strength) is written to its own csv file
#     as soon as it is finished, and the files that already exist are skipped (RESUME).
#
# Output:  data/split/tau<tau>_<alpha|eta>_<strength>.csv   one row per (fraction, gap); columns
#            fraction, gap, t_ini*, rho_r*, delta
#          (fraction = 1 is the single pulse, delta = 0 there; fraction = 0.5 is half and half, 0.9 is "90-10")
#          data/split_summary.csv   one row per (tau, model, strength): the single pulse and the best split
# ---------------------------------------------------------------------------------------------------------
using DifferentialEquations, DelimitedFiles, Printf

# ---- settings ----------------------------------------------------------------------------------------------
const MAX_THREADS = 30
const T_DEAD  = 250.0          # deadline T: the control is off after T (as in Figs. 5-7)
const T_END   = 5000.0         # integration horizon (the final size is converged at this time)
const VALIDATE = !isempty(ARGS) && ARGS[1] == "validate"    # julia --project=. -t 10 splitting/bang_bang_check_split_tau_v2.jl validate
const TAUS    = VALIDATE ? [50.0] : (isempty(ARGS) ? [20.0, 30.0, 50.0, 80.0] : parse.(Float64, ARGS))
const ALPHAS  = collect(0.01:0.05:1.0)                                   # 0.01, 0.06, ..., 0.96
const ETAS    = collect(range(0.01, 3.0; length = length(ALPHAS)))       # same number of values, 0.01 ... 3
const STRENGTHS = VALIDATE ?
    [(true, 0.005), (true, 0.01), (true, 0.5), (false, 0.25), (false, 0.35), (false, 1.0)] :   # the six panels of Fig. 5
    vcat([(true, a) for a in ALPHAS], [(false, e) for e in ETAS])   # (is_alpha, strength)
const OUTDIR  = VALIDATE ? "data/split_validate" : "data/split"
const SUMMARY = VALIDATE ? "data/split_validate_summary.csv" : "data/split_summary.csv"
const FRACTIONS = collect(0.1:0.1:0.9)    # f = tau1/tau (both orders are covered: 0.1 = "10-90", 0.9 = "90-10")
const GAPS      = collect(0.0:5.0:245.0)  # time between the end of pulse 1 and the start of pulse 2
const COARSE, FINE = 2.0, 0.1             # t_ini search: coarse grid on [0, T], then fine grid around the minimum
const RESUME = true

# ---- SIRC model (same equations, parameters and tolerances as export_data.jl) -----------------------------------
const BETA, GAMMA, OMEGA, I0 = 0.2, 0.1, 0.025, 1e-6
const RELTOL = 1e-8
const ABSTOL = [1e-10, 1e-10, 1e-60, 1e-10]    # tiny absolute tolerance on rho_i: keeps it positive between outbreaks

# u = [rho_c, rho_s, rho_i, rho_r]; p = (is_alpha, a): Gamma = a (alpha model) or Gamma = a*rho_i (eta model)
function sirc!(du, u, p, t)
    is_alpha, a = p
    rho_c, rho_s, rho_i, rho_r = u
    Gamma = is_alpha ? a : a * rho_i
    du[1] = Gamma * rho_s - OMEGA * rho_c
    du[2] = OMEGA * rho_c - Gamma * rho_s - BETA * rho_s * rho_i
    du[3] = BETA * rho_s * rho_i - GAMMA * rho_i
    du[4] = GAMMA * rho_i
end

# Integrate from t0 to t1 with a constant control `a`; returns the final state (nothing else is stored).
function advance(u, t0, t1, is_alpha, a)
    t1 <= t0 && return u
    prob = ODEProblem(sirc!, u, (t0, t1), (is_alpha, a))
    sol = solve(prob, Tsit5(); reltol = RELTOL, abstol = ABSTOL, save_everystep = false)
    # a failed integration (e.g. maxiters) must stop the run, never silently give a wrong number
    occursin("Success", string(sol.retcode)) || error("ODE solver failed: retcode = $(sol.retcode) (t0=$t0, t1=$t1, a=$a)")
    return sol.u[end]
end

# Final size for a campaign of strength `a` with up to two pulses:
#   pulse 1 on [t1, t1 + tau1], pulse 2 on [t1 + tau1 + gap, t1 + tau1 + gap + tau2], everything cut at T.
# tau2 = 0 gives the single pulse (the result does not depend on gap then).
function two_pulse_final_size(is_alpha, a, t1, tau1, gap, tau2)
    e1 = min(t1 + tau1, T_DEAD)
    s2 = min(e1 + gap, T_DEAD)
    e2 = min(s2 + tau2, T_DEAD)
    u = [0.0, 1.0 - I0, I0, 0.0]
    u = advance(u, 0.0, t1, is_alpha, 0.0)              # before the first pulse
    u = advance(u, t1, e1, is_alpha, a)                 # pulse 1
    u = advance(u, e1, s2, is_alpha, 0.0)               # gap
    u = advance(u, s2, e2, is_alpha, a)                 # pulse 2
    u = advance(u, max(e2, t1), T_END, is_alpha, 0.0)   # after the campaign, up to T_END
    return u[4]
end

# Optimal start time: coarse grid on [0, T], then a fine grid around the best coarse point.
function optimize_tini(f::Function)
    coarse = collect(0.0:COARSE:T_DEAD)
    vals = [f(t) for t in coarse]
    i = argmin(vals)
    lo, hi = coarse[max(i - 1, 1)], coarse[min(i + 1, length(coarse))]
    fine = collect(lo:FINE:hi)
    fvals = [f(t) for t in fine]
    j = argmin(fvals)
    return fine[j], fvals[j]
end

# Jobs for one value of tau: (fraction, gap). The first one, (1, 0), is the single pulse (baseline). The
# combinations whose second pulse would start after T even with t_ini = 0 are left out (they are a single
# pulse of length f*tau, that is, worse).
function make_jobs(tau)
    jobs = Tuple{Float64,Float64}[(1.0, 0.0)]
    for f in FRACTIONS, g in GAPS
        g <= T_DEAD - f * tau && push!(jobs, (f, g))
    end
    return jobs
end

const INTEGRATIONS_PER_JOB = length(0.0:COARSE:T_DEAD) + 41       # coarse grid + fine grid (about 4 / 0.1 + 1)

csvname(tau, is_alpha, a) = @sprintf("%s/tau%03d_%s_%.3f.csv", OUTDIR, Int(round(tau)), is_alpha ? "alpha" : "eta", a)

# Worker: takes jobs from the shared atomic counter and writes ONLY to its own index j of Topt / Ropt.
function worker!(Topt, Ropt, next, jobs, tau, is_alpha, a)
    n = length(jobs)
    while true
        j = Threads.atomic_add!(next, 1)
        j > n && break
        f, gap = jobs[j]
        tau1, tau2 = f * tau, (1 - f) * tau
        tj, rj = optimize_tini(t -> two_pulse_final_size(is_alpha, a, t, tau1, gap, tau2))
        Topt[j] = tj
        Ropt[j] = rj
    end
    return nothing
end

# One (tau, model, strength): all the jobs are shared dynamically among the threads (atomic counter).
function compute_strength(tau, is_alpha, a, file)
    jobs = make_jobs(tau)
    n = length(jobs)
    Topt = Vector{Float64}(undef, n)
    Ropt = Vector{Float64}(undef, n)
    next = Threads.Atomic{Int}(1)
    # one task per thread; each one runs the worker function below, in which EVERY variable is local
    # (the old draft assigned f, gi, ... inside the threaded block while the same names existed in the enclosing
    # scope: all the threads then shared them -> data race -> wrong numbers).
    Threads.@threads for _ in 1:Threads.nthreads()
        worker!(Topt, Ropt, next, jobs, tau, is_alpha, a)
    end
    Tbase, Rbase = Topt[1], Ropt[1]
    # sanity check: two pulses without gap are the single pulse
    for f in (0.3, 0.7)
        d = abs(two_pulse_final_size(is_alpha, a, Tbase, f * tau, 0.0, (1 - f) * tau) - Rbase)
        d < 1e-6 || @warn "gap = 0 check failed" tau is_alpha a f d
    end
    data = Matrix{Float64}(undef, n, 5)
    for j in 1:n
        data[j, :] = [jobs[j][1], jobs[j][2], Topt[j], Ropt[j], Ropt[j] - Rbase]
    end
    writedlm(file * ".tmp", data, ',')
    mv(file * ".tmp", file; force = true)               # the csv only exists when it is complete
    return nothing
end

# ---- run ---------------------------------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    Threads.nthreads() > MAX_THREADS && error("Use at most $MAX_THREADS threads (julia -t $MAX_THREADS ...)")
    mkpath(OUTDIR)
    njobs = sum(length(make_jobs(tau)) for tau in TAUS) * length(STRENGTHS)
    @printf("Threads: %d | tau = %s | %d strengths (%d alpha + %d eta) | T = %g, t_end = %g\n",
            Threads.nthreads(), string(TAUS), length(STRENGTHS), length(ALPHAS), length(ETAS), T_DEAD, T_END)
    @printf("Jobs (fraction, gap): %d, about %.2e integrations of the model (%d t_ini values each)\n",
            njobs, njobs * INTEGRATIONS_PER_JOB, INTEGRATIONS_PER_JOB)
    t_start = time()
    for tau in TAUS, (is_alpha, a) in STRENGTHS
        file = csvname(tau, is_alpha, a)
        if RESUME && isfile(file)
            continue
        end
        t0 = time()
        compute_strength(tau, is_alpha, a, file)
        GC.gc()                                       # keep the heap small (no swap on the server)
        @printf("tau = %g, %s = %.3f done in %.0f s (total %.1f min)\n", tau, is_alpha ? "alpha" : "eta", a,
                time() - t0, (time() - t_start) / 60)
    end

    # ---- summary: single pulse vs. best split for every (tau, model, strength) --------------------------------
    rows = Vector{Vector{Float64}}()
    for tau in TAUS, (is_alpha, a) in STRENGTHS
        D = readdlm(csvname(tau, is_alpha, a), ',')
        base = D[1, :]                                 # fraction = 1: the single pulse
        S = D[2:end, :]
        b = argmin(S[:, 4])
        push!(rows, [tau, is_alpha ? 1.0 : 0.0, a, base[3], base[4], S[b, 1], S[b, 2], S[b, 3], S[b, 4], S[b, 4] - base[4]])
    end
    open(SUMMARY, "w") do io
        println(io, "tau,is_alpha,strength,t_ini_single,rho_single,best_fraction,best_gap,t_ini_split,rho_split,delta")
        for r in rows
            println(io, join(r, ","))
        end
    end
    order = sortperm([r[10] for r in rows])             # most negative delta first = splitting helps the most
    println("\nLargest differences with respect to the single pulse (delta = rho_split - rho_single):")
    for i in order[1:min(15, length(order))]
        r = rows[i]
        @printf("  tau = %3g  %-5s %.3f : delta = %+.2e   (f = %.1f, gap = %g, t_ini = %.1f; single t_ini = %.1f, rho = %.5f)\n",
                r[1], r[2] == 1.0 ? "alpha" : "eta", r[3], r[10], r[6], r[7], r[8], r[4], r[5])
    end
    println("\nSummary saved in $SUMMARY")
end
