import scipy.stats as stats

f05 = [59.65, 66.12, 63.39, 67.64]
edit_density = [37.6, 32.7, 34.7, 34.7]
oci = [3.50, 2.82, 2.95, 3.03]
other_fp = [23.7, 15.1, 21.3, 36.3]

print("Edit Density vs OCI:", stats.spearmanr(edit_density, oci))
print("OTHER FP vs OCI:", stats.spearmanr(other_fp, oci))
print("F0.5 vs OCI:", stats.spearmanr(f05, oci))