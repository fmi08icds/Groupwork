from regression_edu.models.locally_weighted_regression import LocallyWeightedRegression
from regression_edu.models.linear_regression import LinearRegression
from localreg import localreg
from sklearn.linear_model import LinearRegression as LR
from memory_profiler import profile

import pstats
import cProfile, numpy as np

profile_lwr_ours = cProfile.Profile()
profile_lwr_std = cProfile.Profile()
profile_lin_ours = cProfile.Profile()
profile_lin_std = cProfile.Profile()


def gen_data(size, dim):
    x1 = np.linspace(-100, 100, size)
    x2 = np.linspace(100, -100, size)
    x3 = np.linspace(-50, 50, size)
    if dim == 3:
        yf = [
            0.1 * xi**2 + 3 * xi - 20 * x2[i] + x3[i] ** 3 for i, xi in enumerate(x1)
        ]
    if dim == 2:
        yf = [0.1 * xi**2 + 3 * xi - 20 * x2[i] for i, xi in enumerate(x1)]
    if dim == 1:
        yf = [0.1 * xi**2 + 3 * xi - 20 for i, xi in enumerate(x1)]
    y = np.asarray([yi + np.random.random() * 100 - 50 for yi in yf])
    # return x1,None,None, y
    if dim == 3:
        return x1, x2, x3, y
    if dim == 2:
        return x1, x2, y
    if dim == 1:
        return x1, y


def main():
    lr = LR()
    for i in range(50):
        x1, x2, x3, y = gen_data(50, 3)

        profile_lin_ours.enable()
        # data = np.asarray((x1,y)).T
        LinearRegression([x1, x2, x3, y], transposed=True)
        profile_lin_ours.disable()
        print("Lin Reg done")
        profile_lin_std.enable()
        lr.fit(np.asarray([x1, x2, x3]).T, y)
        # lr.fit(np.asarray([[i] for i in x1]),y)
        profile_lin_std.disable()
        print("Std lin reg done")

        profile_lwr_ours.enable()
        LocallyWeightedRegression([x1, x2, x3, y], transposed=True)
        profile_lwr_ours.disable()
        print("LWR done")
        profile_lwr_std.enable()
        localreg(np.asarray([x1, x2, x3]).T, y)
        profile_lwr_std.disable()
        print("Std lwr done")
        print(i)

    profile_lwr_ours.dump_stats("lwr_ours.stats")
    profile_lwr_std.dump_stats("lwr_standard.stats")
    profile_lin_ours.dump_stats("lin_ours.stats")
    profile_lin_std.dump_stats("lin_standard.stats")


@profile
def mem_our_lin(x, y):
    LinearRegression([x, y], transposed=True)


@profile
def mem_std_lin(x, y):
    lr = LR()
    lr.fit(np.asarray([[i] for i in x]), y)


@profile
def mem_our_lwr(x, y):
    LocallyWeightedRegression([x, y], transposed=True)


@profile
def mem_std_lwr(x, y):
    localreg(x, y)


if __name__ == "__main__":
    main()
    x, y = gen_data(500, 1)
    mem_our_lin(x, y)
    mem_std_lin(x, y)
    mem_our_lwr(x, y)
    mem_std_lwr(x, y)
