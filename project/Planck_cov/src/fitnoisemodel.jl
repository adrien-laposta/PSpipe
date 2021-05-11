#
# ```@setup fitnoisemodel
# # the example command line input for this script
# ARGS = ["example.toml", "100", "EE", "--plot"] 
# ``` 

configfile, freq, spec = ARGS

# # [Fit Noise Model (fitnoisemodel.jl)](@id fitnoisemodel)

## setup data
using Plots
using TOML
using BlackBoxOptim
include("util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
run_name = config["general"]["name"]
spectrapath = joinpath(config["dir"]["scratch"], "rawspectra")
XY = Symbol(spec)
lmax = min(2508,nside2lmax(nside))


# Next, we check to see if we need to render plots for the Documentation.

if "--plot" ∉ ARGS
    Plots.plot(args...; kwargs...) = nothing
    Plots.plot!(args...; kwargs...) = nothing
end

# We read in the raw spectra generated from rawspectra.jl.

Cl11 = DataFrame(CSV.File(joinpath(spectrapath,"$(run_name)_P$(freq)hm1xP$(freq)hm1.csv")));
Cl12 = DataFrame(CSV.File(joinpath(spectrapath,"$(run_name)_P$(freq)hm1xP$(freq)hm2.csv")));
Cl22 = DataFrame(CSV.File(joinpath(spectrapath,"$(run_name)_P$(freq)hm2xP$(freq)hm2.csv")));

truncate(vec::Vector, lmax) = SpectralVector(vec[firstindex(vec):(lmax+1)])
truncate(vec::SpectralVector, lmax) = vec[IdentityRange(firstindex(vec):lmax)]

cl11 = truncate(Cl11[!,XY], lmax)
cl12 = truncate(Cl12[!,XY], lmax)
cl22 = truncate(Cl22[!,XY], lmax)

Wl11 = util_planck_beam_Wl(freq, "hm1", freq, "hm1", XY, XY; 
    lmax=lmax, beamdir=config["dir"]["beam"])
Wl12 = util_planck_beam_Wl(freq, "hm1", freq, "hm2", XY, XY; 
    lmax=lmax, beamdir=config["dir"]["beam"])
Wl22 = util_planck_beam_Wl(freq, "hm2", freq, "hm2", XY, XY; 
    lmax=lmax, beamdir=config["dir"]["beam"])

Wl11, Wl12, Wl22 = map(v->truncate(v,lmax), (Wl11, Wl12, Wl22))

cl = cl12 ./ Wl12
nl1 = (cl11 ./ Wl11 .- cl) .* Wl11
nl2 = (cl22 ./ Wl22  .- cl) .* Wl22;

ell = eachindex(cl)
plot(ell, ell.^2 .* cl12, label="unbeamed $(run_name) $(freq) $(spec)",
    xlabel="multipole moment", ylabel="\$\\ell^2 C_{\\ell}^{$spec}\$", xlim=(0,lmax))

#

@. camspec_model(ℓ, α) =  
    α[1] * (100. / ℓ)^α[2] + α[3] * (ℓ / 1000.)^α[4] / ( 1 + (ℓ / α[5])^α[6] )^α[7]

# 

function fit_bb_model(model, p0, xl, yl, signal; kwargs...)
    lower = map(x -> x-0.9abs(x), p0)
    upper = map(x -> x+0.9abs(x), p0)
    like(α) = (sum((2 .* xl .+ 1) ./ (model(xl, p0).^2 .+ signal.^2) .* (model(xl, α) .- yl).^2))
    println("starting opt ", like(p0))
    res = bboptimize(like; SearchRange=map((i,j)->(i,j), lower, upper), NumDimensions = length(p0), 
        MaxFuncEvals=50_000, TraceInterval=20)
    return best_candidate(res)
end

# 
p0_1 = readdlm(joinpath(config["dir"]["pspipe_project"], "output", 
        "planck_noise_coeffs", "$(freq)_hm1_$(spec)_coeff.dat"))[:,1]
p0_2 = readdlm(joinpath(config["dir"]["pspipe_project"], "output", 
        "planck_noise_coeffs", "$(freq)_hm2_$(spec)_coeff.dat"))[:,1]

#
min_ell_ind = 31 # lmin=30
pfit_1 = fit_bb_model(camspec_model, p0_1, 
    parent(ell)[min_ell_ind:end], parent(nl1)[min_ell_ind:end], parent(cl)[min_ell_ind:end])
pfit_2 = fit_bb_model(camspec_model, p0_2, 
    parent(ell)[min_ell_ind:end], parent(nl2)[min_ell_ind:end], parent(cl)[min_ell_ind:end])

#

mean(x) = sum(x) / length(x)
plot((nl1), alpha=0.5, label="nl hm1")
plot!(2:lmax, [camspec_model(ℓ, p0_1) for ℓ in 2:lmax], ylim=(0.0, 4mean(abs.(nl1))), label="initial model")
plot!(2:lmax, [camspec_model(ℓ, pfit_1) for ℓ in 2:lmax], label="fitted model", linestyle=:dash)

#
plot((nl2), alpha=0.5, label="nl hm2")
plot!(2:lmax, [camspec_model(ℓ, p0_2) for ℓ in 2:lmax], ylim=(0.0, 2mean(nl2)), label="initial model")
plot!(2:lmax, [camspec_model(ℓ, pfit_2) for ℓ in 2:lmax], ylim=(0.0, 2mean(nl2)), label="fitted model", linestyle=:dash)

#
coefficientpath = joinpath(config["dir"]["scratch"], "noise_model_coeffs")
mkpath(coefficientpath)

open(joinpath(coefficientpath, "$(run_name)_$(freq)_$(spec)_hm1.dat"), "w") do io
   writedlm(io, pfit_1)
end
open(joinpath(coefficientpath, "$(run_name)_$(freq)_$(spec)_hm2.dat"), "w") do io
   writedlm(io, pfit_2)
end
