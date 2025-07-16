# %% Percents of mismatch between cavities
from finesse.ligo.lho import make_O4_lho
lho = make_O4_lho()
print(lho.cavity_mismatches_table(percent=True, numfmt='{:.2g}')[0])
print(lho.cavity_mismatches_table(percent=True, numfmt='{:.2g}')[1])

# %% Print where the mismatches happen
print(lho.mismatches_table())
# %%
