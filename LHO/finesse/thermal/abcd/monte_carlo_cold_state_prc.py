# %%
import finesse.ligo
from finesse.ligo.factory import ALIGOFactory

import finesse.materials
import numpy as np
import matplotlib.pyplot as plt
import cmath

finesse.init_plotting(fmts=['png'], dpi=120)

def gouy_rt(RT):
    B = RT[0, 1]
    A = RT[0, 0] 
    D = RT[1, 1]
    return np.rad2deg(np.sign(B)*np.arccos((A+D)/2))

def symbolic_lambdify(symbol, args=None):
    sym_str = str(symbol)
    if args is None:
        args = symbol.parameters()
    ARGS = []
    for i, arg in enumerate(args):
        ARGS.append(arg.full_name.replace('.', '__'))
        sym_str = sym_str.replace(arg.full_name, ARGS[-1])
    return eval(f"lambda {','.join(ARGS)}: {sym_str}")

def symbolic_vectored_lambdify(symbol, args=None):
    sym_str = str(symbol)
    if args is None:
        args = symbol.parameters()
    ARGS = []
    for i, arg in enumerate(args):
        ARGS.append(f'x[{i}, None]')
        sym_str = sym_str.replace(arg.full_name, ARGS[-1])
    return eval(f"lambda x: {sym_str}")

def compute_q(A, B, C, D):
    if C == 0.0:  # confocal cavity - g = 0 (critical)
        return None
    half_inv_C = 0.5 / C

    D_minus_A = D - A
    minus_B = -B

    sqrt_term = cmath.sqrt(D_minus_A * D_minus_A - 4 * C * minus_B)
    lower = (-D_minus_A - sqrt_term) * half_inv_C
    upper = (-D_minus_A + sqrt_term) * half_inv_C

    if lower.imag > 0:
        q = lower
    elif upper.imag > 0:
        q = upper
    else:
        return None

    return q

ITM_sub = 300.39e-6
ITM_srf = -46.52e-6
ETM_sub = 209.57e-6
ETM_srf = -33.46e-6
IRH_sub = -9.92e-6
IRH_srf = 0.992e-6
ERH_sub = -12.53e-6
ERH_srf = 0.841e-6

# We first make a factory object that can generate an ALIGO model
# here we do so using the LHO O4 parameter file
factory = ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")

# %%
factory.reset()
factory.options.INPUT.add_IMC_and_IM1 = True # include the IMC cavity model
lho = factory.make()
base = lho.deepcopy() # copy to grab defaults from later

# %%
def generate(direction, symbols):
    RT1 = lho.propagate_beam(
        lho.PRM.p1.o,
        lho.PRM.p1.i,
        lho.ITMX.p2.o,
        symbolic=symbols,
        simplify=True,
        direction=direction,
    )
    RT2 = lho.propagate_beam(
        lho.PRM.p1.i,
        lho.PRM.p1.o,
        symbolic=symbols,
        simplify=True,
        direction=direction,
    )
    #global ABCD_RT, fA, fB, fC, fD
    ABCD_RT = RT1.abcd() @ RT2.abcd()

    fA = symbolic_vectored_lambdify(ABCD_RT[0, 0], symbols)
    fB = symbolic_vectored_lambdify(ABCD_RT[0, 1], symbols)
    fC = symbolic_vectored_lambdify(ABCD_RT[1, 0], symbols)
    fD = symbolic_vectored_lambdify(ABCD_RT[1, 1], symbols)

    return lambda x: gouy_rt(np.array([
        [
            fA(x),
            fB(x)
        ],
        [
            fC(x),
            fD(x)
        ],
    ]))

x_symbols = (lho.PRM.Rcx.ref, lho.PR3.Rcx.ref, lho.PR2.Rcx.ref, lho.lp1.L.ref, lho.lp2.L.ref)
y_symbols = (lho.PRM.Rcy.ref, lho.PR3.Rcy.ref, lho.PR2.Rcy.ref, lho.lp1.L.ref, lho.lp2.L.ref)

RX_RT_GOUYx = generate('x', x_symbols)
RX_RT_GOUYy = generate('y', y_symbols)

# %%
x = np.array(
    (
    lho.PRM.Rcx.value,
    lho.PR3.Rcx.value,
    lho.PR2.Rcx.value,
    lho.lp1.L.value,
    lho.lp2.L.value,
    )
)

RX_RT_GOUYx(x), RX_RT_GOUYy(x)
# %%
mean = np.array((
    lho.PRM.Rcx.value,
    lho.PR3.Rcx.value,
    lho.PR2.Rcx.value,
    lho.lp1.L.value,
    lho.lp2.L.value,
))

# Can try uniform sampling, essentially no knowledge of what the parameters are
std = np.array((
    0.002,
    5e-3, # ??
    5e-3, # 
    0.01,
    0.01,
))
samples = np.random.normal(mean, std, size=(1000000, len(mean))).T

# Can try uniform sampling, essentially no knowledge of what the parameters are
# std = np.array((
#     0.1,
#     0.05,
#     0.1,
#     0.02,
#     0.02,
# ))
# samples = np.random.uniform(mean-std, mean+std, size=(2000000, len(mean))).T

# %%
samples_gouy_x = RX_RT_GOUYx(samples).squeeze()
samples_gouy_y = RX_RT_GOUYy(samples).squeeze()
result = np.vstack([samples, samples_gouy_x, samples_gouy_y])
notnan = ~np.isnan(result.sum(0))
result = result[:, notnan]
# %%
mean_gouy = result[-2:].mean(0)
idx = np.bitwise_and(
    # two because we measured about 20 single pass, and this
    # is comparing roundtrip
    result[-2:].mean(0) > (20.6-0.3)*2,
    result[-2:].mean(0) < (20.6+0.3)*2,
)
# %%
import corner
figure = corner.corner(
    result.T[idx, :-2],
    labels=[
        r"$Rc PRM$",
        r"$Rc PR3$",
        r"$Rc PR2$",
        r"$L, PRM \rightarrow 2$",
        r"$L, PR2 \rightarrow 3$",
        r"X",
        r"Y",
    ],
    quantiles=[0.16, 0.5, 0.84],
    title_kwargs={"fontsize": 12},
    plot_contours=False,
    show_titles=True,
    smooth=1,
    bins=100,
)

# Extract the axes
ndim = 5
axes = np.array(figure.axes).reshape((ndim, ndim))

# Loop over the diagonal
for i in range(5):
    ax = axes[i, i]
    ax.axvline(mean[i], color="r", lw=3)

# Loop over the histograms
for yi in range(5):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.plot(mean[xi], mean[yi], "sr")

plt.savefig("./figures/cold_state_mcmc_prc.pdf")
# %%
means = result.mean(1)
stds = result.std(1)

for a in zip(x_symbols, means, stds):
    print(a[0].full_name, f"{a[1]:.5f}", f"{a[2]:.3f}")
