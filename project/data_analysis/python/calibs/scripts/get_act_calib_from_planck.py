from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import sys, os
import re
import pickle
from cobaya.run import run
from getdist.mcsamples import loadMCSamples
import matplotlib.pyplot as plt

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


spec_dir = "spectra_pk/"
cov_dir = "covariances_pk/"

output_dir = "output_calib/"
pspy_utils.create_directory(output_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

# Multipoole range to use
lmin, lmax = 1000, 2000

# Set planck reference maps
reference_maps = {"s19_pa4_f220": "Planck_f217",
                  "s19_pa6_f090": "Planck_f100",
                  "s19_pa6_f150": "Planck_f143"}

spectra_and_covariances = {}
for act_map in reference_maps:
    print("Calibration of %s ..." % act_map)
    planck_map = reference_maps[act_map]
    print("Using %s map ..." % planck_map)

    lb, ps_actxact = so_spectra.read_ps(
            spec_dir + "Dl_%sx%s_cross.dat"%(act_map, act_map),
            spectra = spectra)
    try:
        cross_name = "%sx%s" % (act_map, planck_map)
        lb, ps_actxplanck = so_spectra.read_ps(
                spec_dir + "Dl_%s_cross.dat"%(cross_name),
                spectra = spectra)
    except:
        cross_name = "%sx%s" % (planck_map, act_map)
        lb, ps_actxplanck = so_spectra.read_ps(
                spec_dir + "Dl_%s_cross.dat"%(cross_name),
                spectra = spectra)
    print("ACTxPlanck : %s" % cross_name)
    cov = {}
    ref_name = "%sx%s"%(act_map, act_map)
    cov["refxref"] = np.load(
            cov_dir + "analytic_cov_{0}_{0}.npy".format(ref_name))
    cov["crossxcross"] = np.load(
            cov_dir + "analytic_cov_{0}_{0}.npy".format(cross_name))
    try:
        cov["refxcross"] = np.load(
                cov_dir + "analytic_cov_{0}_{1}.npy".format(ref_name, cross_name))
        print("Cross covariance : %s_%s" % (ref_name, cross_name))
    except:
        cov["refxcross"] = np.load(
                cov_dir + "analytic_cov_{0}_{1}.npy".format(cross_name, ref_name))
        print("Cross covariance : %s_%s" % (cross_name, ref_name))
    TT_actxact = ps_actxact["TT"]
    TT_actxplanck = ps_actxplanck["TT"]
    id = np.where((lb>=lmin)&(lb<=lmax))
    TT_actxact = TT_actxact[id]
    TT_actxplanck = TT_actxplanck[id]
    TT = {"actxact": TT_actxact, "actxplanck": TT_actxplanck}
    for key in cov:
        cov[key] = so_cov.selectblock(cov[key], modes,
                                      n_bins = len(lb),
                                      block = "TTTT")
        cov[key] = cov[key][np.ix_(id[0], id[0])]
    spectra_and_covariances[act_map] = (TT, cov)

for act_map in reference_maps:
    TT, TT_cov = spectra_and_covariances[act_map]

    def loglike(c):
        res_ps = TT["actxact"] - (1/c) * TT["actxplanck"]
        res_cov = (TT_cov["refxref"] + (1/c)**2 * TT_cov["crossxcross"]
                    - 2 * (1/c) * TT_cov["refxcross"])
        chi2 = res_ps @ np.linalg.solve(res_cov, res_ps)
        logL = -0.5 * chi2
        logL -= len(TT["actxact"]) / 2 * np.log(2 * np.pi)
        logL -= 0.5 * np.linalg.slogdet(res_cov)[1]

        return(logL)

    info = {
        "likelihood": {"my_like": loglike},
        "params": {
            "c": {"prior": {"min": 0.5, "max": 1.5}, "latex": r"c_{%s}"%act_map}
                },
        "sampler": {
            "mcmc": {
                "max_tries": 1e4,
                "Rminus1_stop": 0.0005
                    }
                   },
        "output": "%s/%s/mcmc"%(output_dir, act_map),
        "force": True
            }

    updated_info, sampler = run(info)


# Chain analysis

out_cal_dict = {}

fig, axes = plt.subplots(1, 3, figsize = (15, 5))

for i, act_map in enumerate(reference_maps):

    chains = "%s/%s/mcmc" % (output_dir, act_map)
    samples = loadMCSamples(chains, settings = {"ignore_rows": 0.5})
    mean_calib = samples.getMeans(pars = [0])[0]
    std_calib = np.sqrt(samples.getCovMat().matrix[0, 0])
    out_cal_dict[act_map] = [mean_calib, std_calib]
    cal_post = samples.get1DDensity('c')
    x = np.linspace(mean_calib - 4 * std_calib,
                    mean_calib + 4 * std_calib,
                    100)
    y = cal_post.Prob(x)
    axes[i].grid(True, ls = "dotted")
    axes[i].plot(x, y, color = "tab:red", lw = 2)
    axes[i].set_xlabel(r"$c\_{%s}$"%act_map.replace("_", "\_"), fontsize = 12)

plt.tight_layout()
plt.savefig("%s/act_planck_calib.pdf"%output_dir)
pickle.dump(out_cal_dict, open("%s/act_planck_calib_dict.pkl"%output_dir, "wb"))
