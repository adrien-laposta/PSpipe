'''
This script is used to compute the mode coupling matrices of the Planck data.
The inputs for the script are the Planck beam and likelihood masks.
To run it:
python get_planck_mcm_Bbl.py global.dict
'''
import numpy as np
import healpy as hp
from pspy import so_dict, so_map, so_mcm, pspy_utils, so_mpi
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcm_dir = "mcms"

pspy_utils.create_directory(windows_dir)
pspy_utils.create_directory(mcm_dir)

freqs = d["freqs"]
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
pixwin = d["pixwin"]
splits = d["splits"]
experiment = "Planck"

print("Compute Planck 2018 mode coupling matrices")

freq1_list, hm1_list, freq2_list, hm2_list = [], [], [], []
n_mcms = 0
for f1, freq1 in enumerate(freqs):
    for count1, hm1 in enumerate(splits):
        for f2, freq2 in enumerate(freqs):
            if f1 > f2: continue
            for count2, hm2 in enumerate(splits):
                if (count1 > count2) & (f1 == f2): continue
                
                freq1_list += [freq1]
                freq2_list += [freq2]
                hm1_list += [hm1]
                hm2_list += [hm2]
                n_mcms += 1
                
print("number of mcm matrices to compute : %s" % n_mcms)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_mcms - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    freq1, hm1, freq2, hm2 = freq1_list[task], hm1_list[task], freq2_list[task], hm2_list[task]
    

    win_t1 = so_map.read_map("%s/window_T_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))
    win_pol1 = so_map.read_map("%s/window_P_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))

    window_tuple1 = (win_t1, win_pol1)
    

    win_t2 = so_map.read_map("%s/window_T_%s_%s-%s.fits" % (windows_dir, experiment, freq2, hm2))
    win_pol2 = so_map.read_map("%s/window_P_%s_%s-%s.fits" % (windows_dir, experiment, freq2, hm2))

    window_tuple2 = (win_t2, win_pol2)

        
    del win_t1, win_pol1
    
    l, bl1_t = np.loadtxt(d["beam_%s_%s_T" % (freq1, hm1)], unpack=True)
    l, bl1_pol = np.loadtxt(d["beam_%s_%s_pol" % (freq1, hm1)], unpack=True)

    if pixwin == True:
        bl1_t *= hp.pixwin(window_tuple1[0].nside)[:len(bl1_t)]
        bl1_pol *= hp.pixwin(window_tuple1[0].nside)[:len(bl1_pol)]
        
    bl_tuple1 = (bl1_t, bl1_pol)


    del win_t2, win_pol2
                
    l, bl2_t = np.loadtxt(d["beam_%s_%s_T" % (freq2, hm2)], unpack=True)
    l, bl2_pol = np.loadtxt(d["beam_%s_%s_pol" % (freq2, hm2)], unpack=True)

    if pixwin == True:
        bl2_t *= hp.pixwin(window_tuple2[0].nside)[:len(bl2_t)]
        bl2_pol *= hp.pixwin(window_tuple2[0].nside)[:len(bl2_pol)]

    bl_tuple2 = (bl2_t, bl2_pol)
                
    mcm_inv, mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=window_tuple1,
                                                         win2=window_tuple2,
                                                         binning_file=binning_file,
                                                         bl1=bl_tuple1,
                                                         bl2=bl_tuple2,
                                                         lmax=lmax,
                                                         niter=niter,
                                                         type=type,
                                                         unbin=True,
                                                         save_file="%s/%s_%sx%s_%s-%sx%s" % (mcm_dir, experiment, freq1, experiment, freq2, hm1, hm2))




