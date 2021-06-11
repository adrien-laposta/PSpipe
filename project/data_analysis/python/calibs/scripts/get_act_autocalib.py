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

surveys = d["surveys"]
arrays_s17 = d["arrays_s17"]
arrays_s18 = d["arrays_s18"]
arrays_s19 = d["arrays_s19"]

arrays = {"s17": arrays_s17,
          "s18": arrays_s18,
          "s19": arrays_s19}

spec_dir = "spectra/"
cov_dir = "covariances/"

output_dir = "output_calib/"
pspy_utils.create_directory(output_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

# Multipoole range to use
lmin, lmax = 1000, 2000

# Map for which we set cal = 1.0
reference_maps = {90: "s19_pa6_f090",
                  150: "s19_pa6_f150",
                  220: "s19_pa4_f220"}
calib_dict = {}
for surv in surveys:
    for arr in arrays[surv]:
        calib_dict["%s_%s"%(surv, arr)] = {"cal": d["cal_%s_%s"%(surv, arr)],
                                           "freq": d["nu_eff_%s_%s"%(surv, arr)]}

spectra_and_covariances = {}

for cal_name in calib_dict:
    freq = calib_dict[cal_name]["freq"]
    ref_array = reference_maps[freq]

    lb, ps_ref = so_spectra.read_ps(
        spec_dir + "Dl_%sx%s_cross.dat"%(ref_array, ref_array),
        spectra = spectra)
    try:
        cross_name = "%sx%s"%(cal_name, ref_array)
        lb, ps_cross = so_spectra.read_ps(
            spec_dir + "Dl_%s_cross.dat"%(cross_name),
            spectra = spectra)

    except:
        cross_name = "%sx%s"%(ref_array, cal_name)
        lb, ps_cross = so_spectra.read_ps(
            spec_dir + "Dl_%s_cross.dat"%(cross_name),
            spectra = spectra)

    cov = {}

    ref_name = "%sx%s"%(ref_array, ref_array)
    cov["refxref"] = np.load(
        cov_dir + "analytic_cov_{0}_{0}.npy".format(ref_name))

    cov["crossxcross"] = np.load(
        cov_dir + "analytic_cov_{0}_{0}.npy".format(cross_name))

    try:
        cov["refxcross"] = np.load(
            cov_dir + "analytic_cov_{0}_{1}.npy".format(ref_name, cross_name))
    except:
        cov["refxcross"] = np.load(
            cov_dir + "analytic_cov_{0}_{1}.npy".format(cross_name, ref_name))

    TT_ref = ps_ref["TT"]
    TT_cross = ps_cross["TT"]
    id = np.where((lb >= lmin) & (lb <= lmax))
    TT_ref = TT_ref[id]
    TT_cross = TT_cross[id]
    TT = {"ref": TT_ref, "cross": TT_cross}

    for key in cov:
        cov[key] = so_cov.selectblock(cov[key], modes,
                                      n_bins = len(lb),
                                      block = "TTTT")
        cov[key] = cov[key][np.ix_(id[0], id[0])]

    spectra_and_covariances[cal_name] = (TT, cov)

for calib in calib_dict:
    if not(calib in reference_maps.values()):

        TT, TT_cov = spectra_and_covariances[calib]

        # log-like for cobaya sampling
        def loglike(c):

            res_ps = TT["ref"] - c * TT["cross"]
            res_cov = (TT_cov["refxref"] + pow(c, 2) * TT_cov["crossxcross"]
                       - 2 * c * TT_cov["refxcross"])
            chi2 = res_ps @ np.linalg.solve(res_cov, res_ps)

            logL = -0.5 * chi2
            logL -= len(TT["ref"]) / 2 * np.log(2 * np.pi)
            logL -= 0.5 * np.linalg.slogdet(res_cov)[1]

            return(logL)

        info = {
            "likelihood": {"my_like": loglike},
            "params": {
                "c": {"prior": {"min": 0.5, "max": 1.5}, "latex": r"c_{%s}"%calib}
                      },
            "sampler": {
                "mcmc": {
                    "max_tries": 1e4,
                    "Rminus1_stop": 0.0005
                        }
                       },
            "output": "%s/%s/mcmc"%(output_dir,calib),
            "force": True
               }
        updated_info, sampler = run(info)

# Chain analysis

out_cal_dict = {}

fig, axes = plt.subplots(3, 5, figsize = (12, 6))
i = 0

for calib in calib_dict:

    if not (calib in reference_maps.values()):
        chains = "%s/%s/mcmc" % (output_dir, calib)
        samples = loadMCSamples(chains, settings = {"ignore_rows": 0.5})
        mean_calib = samples.getMeans(pars = [0])[0]
        std_calib = np.sqrt(samples.getCovMat().matrix[0, 0])

        out_cal_dict[calib] = [mean_calib, std_calib]

        cal_post = samples.get1DDensity('c')
        x = np.linspace(mean_calib - 4 * std_calib,
                        mean_calib + 4 * std_calib,
                        100)
        y = cal_post.Prob(x)

        axes[i//5, i%5].grid(True, ls = "dotted")
        axes[i//5, i%5].plot(x, y, color = "tab:red", lw = 2)
        axes[i//5, i%5].set_xlabel(r"$c\_{%s}$"%calib.replace("_", "\_"), fontsize = 12)

        i += 1
    else:
        out_cal_dict[calib] = [1.0, 0.0]

plt.tight_layout()
plt.savefig("%s/act_auto_calib.pdf"%output_dir)

pickle.dump(out_cal_dict, open("%s/act_auto_calib_dict.pkl"%output_dir, "wb"))
