import matplotlib
from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys, os
from itertools import combinations
from itertools import product

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
n_surveys = len(surveys)
arrays = [d["arrays_%s" % survey] for survey in surveys]

spec_dir = "../../spectra/"
cov_dir = "../../covariances/"

output_dir = "../outputs/"
output_plot_dir = os.path.join(output_dir, "plots/")
output_data_dir = os.path.join(output_dir, "data/")
pspy_utils.create_directory(output_plot_dir)
pspy_utils.create_directory(output_data_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

lmax = d["lmax"]
l_pivot = 1000
lranges_labels = ["full", "low", "high"]
lranges = [[0, lmax], [0, l_pivot], [l_pivot, lmax]]

#### SEASON NULL TESTS ####
if n_surveys < 2:
    sys.exit("Cannot make season null tests with only 1 season.")

# cross spectra name list
cross_spectra = []
for i1, surv1 in enumerate(surveys):
    for j1, arr1 in enumerate(arrays[i1]):
        for i2, surv2 in enumerate(surveys):
            if i2 < i1: continue
            for j2, arr2 in enumerate(arrays[i2]):
                if j2 < j1 and i1 == i2: continue
                cross_spectra.append("%s-%sx%s-%s" % (surv1, arr1, surv2, arr2))

# cross arrays name list
arrays_full_list = np.concatenate(arrays)
cross_array_dict = {"%sx%s" % (arr1, arr2): [] for (
                        arr1, arr2) in product(arrays_full_list, repeat=2)}

cross_ps_dict = {}
for xspec in cross_spectra:
    surv_arr1, surv_arr2 = xspec.split("x")
    surv1, arr1 = surv_arr1.split("-")
    surv2, arr2 = surv_arr2.split("-")
    cross_array_dict["%sx%s" % (arr1, arr2)].append(xspec)

    lb, ps = so_spectra.read_ps(
                  spec_dir + "Dl_%s_%sx%s_%s_cross.dat" % (
                  surv1, arr1, surv2, arr2), spectra = spectra)
    cross_ps_dict[surv1, arr1, surv2, arr2] = [lb, ps]

ps_dict = {}
residual_ps_dict = {}
chi2_array = []
chi2_dict = {}

for xarray in cross_array_dict:
    diff_spectra_list = [m for m in combinations(cross_array_dict[xarray], 2)]
    # Compute every A-B differences for xarray
    for diff_spectra in diff_spectra_list:

        chi2_list = []
        ndof_list = []

        xspec_A, xspec_B = diff_spectra

        surv_arr_A1, surv_arr_A2 = xspec_A.split("x")
        surv_A1, arr_A1 = surv_arr_A1.split("-")
        surv_A2, arr_A2 = surv_arr_A2.split("-")

        surv_arr_B1, surv_arr_B2 = xspec_B.split("x")
        surv_B1, arr_B1 = surv_arr_B1.split("-")
        surv_B2, arr_B2 = surv_arr_B2.split("-")

        # Read power spectra

        lb, psA = cross_ps_dict[surv_A1, arr_A1, surv_A2, arr_A2]

        lb, psB = cross_ps_dict[surv_B1, arr_B1, surv_B2, arr_B2]

        Nbins = len(lb)

        # Load covariance matrices
        covAA = np.load(cov_dir + "analytic_cov_%s_%sx%s_%s_%s_%sx%s_%s.npy" % (
            surv_A1, arr_A1, surv_A2, arr_A2, surv_A1, arr_A1, surv_A2, arr_A2))

        covBB = np.load(cov_dir + "analytic_cov_%s_%sx%s_%s_%s_%sx%s_%s.npy" % (
            surv_B1, arr_B1, surv_B2, arr_B2, surv_B1, arr_B1, surv_B2, arr_B2))

        covAB = np.load(cov_dir + "analytic_cov_%s_%sx%s_%s_%s_%sx%s_%s.npy" % (
            surv_A1, arr_A1, surv_A2, arr_A2, surv_B1, arr_B1, surv_B2, arr_B2))

        # Save power spectra and errors
        stdAA = { mode : np.sqrt(so_cov.selectblock(covAA, modes,
                            n_bins = Nbins, block = mode + mode).diagonal())
                  for mode in modes}

        stdBB = { mode : np.sqrt(so_cov.selectblock(covBB, modes,
                            n_bins = Nbins, block = mode + mode).diagonal())
                  for mode in modes}

        ps_dict[surv_A1, arr_A1, surv_A2, arr_A2] = [lb, psA, stdAA]
        ps_dict[surv_B1, arr_B1, surv_B2, arr_B2] = [lb, psB, stdBB]

        # Compute residuals and errors
        res_cov = covAA + covBB - 2 * covAB
        res_ps_d = { mode : psA[mode] - psB[mode] for mode in modes}

        # Save residuals and errors
        std_res = { mode : np.sqrt(so_cov.selectblock(res_cov, modes,
                               n_bins = Nbins, block = mode + mode).diagonal())
                    for mode in modes}
        residual_ps_dict[surv_A1, arr_A1, surv_A2, arr_A2,
                         surv_B1, arr_B1, surv_B2, arr_B2] = [lb, res_ps_d, std_res]

        for i, lr in enumerate(lranges):

            id = np.where((lb >= lr[0]) & (lb <= lr[1]))
            res_ps = np.concatenate([res_ps_d[mode][id] for mode in modes])

            cut_res_cov = []
            for mode1 in modes:
                line = []
                for mode2 in modes:
                    line.append(
                        so_cov.selectblock(
                            res_cov, modes, n_bins = Nbins,
                            block = mode1 + mode2
                            )[np.ix_(id[0], id[0])]
                        )
                cut_res_cov.append(line)
            cut_res_cov = np.block(cut_res_cov)
            print(cut_res_cov.shape)

            # Compute and save chi2
            chi2 = res_ps @ np.linalg.solve(cut_res_cov, res_ps)
            chi2_dict["all", surv_A1, arr_A1, surv_A2, arr_A2,
                      surv_B1, arr_B1, surv_B2, arr_B2] = [chi2, len(res_ps)]
            chi2_list.append(chi2)
            ndof_list.append(len(res_ps))



        for mode in modes:
            res_covblock = so_cov.selectblock(res_cov, modes,
                                              n_bins = Nbins,
                                              block = mode + mode)
            res_ps = res_ps_d[mode]

            # Compute chi2
            for i, lr in enumerate(lranges):
                id = np.where((lb >= lr[0]) & (lb <= lr[1]))
                cut_ps = res_ps[id]
                cut_covblock = res_covblock[np.ix_(id[0], id[0])]
                chi2 = cut_ps @ np.linalg.solve(cut_covblock, cut_ps)
                chi2_dict[mode, surv_A1, arr_A1, surv_A2, arr_A2,
                          surv_B1, arr_B1, surv_B2, arr_B2,
                          lranges_labels[i]] = [chi2, len(id[0])]
                chi2_list.append(chi2)
                ndof_list.append(len(id[0]))

        chi2_array.append(chi2_list)

# Save dicts
pickle.dump(chi2_dict, open(os.path.join(output_data_dir, "chi2.pkl"), "wb"))
pickle.dump(ps_dict, open(os.path.join(output_data_dir, "spectra.pkl"), "wb"))
pickle.dump(residual_ps_dict, open(os.path.join(output_data_dir, "residuals.pkl"), "wb"))

# Histogram plots

lranges = np.array(lranges)
low = (np.min(lb), lranges[1,1])
high = (lranges[2,0], lmax)


labels = np.array(["TT+TE+EE", "TT", "TE", "ET", "EE",
                  "TT+TE+EE ($%d\leq\ell\leq%d$)" % low,
                  "TT ($%d\leq\ell\leq%d$)" % low,
                  "TE ($%d\leq\ell\leq%d$)" % low,
                  "ET ($%d\leq\ell\leq%d$)" % low,
                  "EE ($%d\leq\ell\leq%d$)" % low,
                  "TT+TE+EE ($%d\leq\ell\leq%d$)" % high,
                  "TT ($%d\leq\ell\leq%d$)" % high,
                  "TE ($%d\leq\ell\leq%d$)" % high,
                  "ET ($%d\leq\ell\leq%d$)" % high,
                  "EE ($%d\leq\ell\leq%d$)" % high])

labels = labels.reshape(3, 5)

chi2_array = np.array(chi2_array)
fig, axes = plt.subplots(3, 5, figsize = (20, 12))

for i in range(len(chi2_array[0])):
    axes[i%3, i//3].hist(chi2_array[:, i], label = labels[i%3, i//3])
    axes[i%3, i//3].axvline(ndof_list[i], ymin = -1, ymax = +2,
                            color = "k", ls = "--")
    axes[i%3, i//3].legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir + "chi2_season_null.pdf"))
