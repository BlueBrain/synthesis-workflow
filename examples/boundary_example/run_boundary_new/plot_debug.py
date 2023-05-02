import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("data.csv", header=None)

    d = df[df[6] < 0]
    dd = df[df[6] > 0]

    plt.figure()
    plt.scatter(d[0], d[1], c="r", s=0.02, marker=".", rasterized=True)
    plt.scatter(dd[0], dd[1], c=dd[6], s=0.02, marker=".", rasterized=True)
    plt.axis("equal")
    plt.colorbar()
    plt.savefig("dist_xy.pdf")

    plt.figure()
    plt.scatter(d[2], d[1], c="r", s=0.02, marker=".", rasterized=True)
    plt.scatter(dd[2], dd[1], c=dd[6], s=0.02, marker=".", rasterized=True)
    plt.axis("equal")
    plt.colorbar()
    plt.savefig("dist_zy.pdf")

    plt.figure()
    plt.scatter(d[0], d[2], c="r", s=0.02, marker=".", rasterized=True)
    plt.scatter(dd[0], dd[2], c=dd[6], s=0.02, marker=".", rasterized=True)
    plt.axis("equal")
    plt.colorbar()
    plt.savefig("dist_xz.pdf")

    n_steps = 200
    n = int(len(df) / n_steps)
    for step in tqdm(range(n_steps + 1)):
        d = df.loc[: step * (n + 1)]

        _d = d[d[6] < 0]
        _dd = d[d[6] > 0]
        plt.figure()
        plt.scatter(_d[0], _d[1], c="r", s=0.01, marker=".", rasterized=True)
        plt.scatter(
            _dd[0], _dd[1], c=_dd[6], s=0.01, marker=".", rasterized=True, vmin=0, vmax=1000
        )
        plt.suptitle(f"computational time [a.u]: {step}")
        plt.colorbar(shrink=0.5, label="distance to boundary [microns]")
        plt.axis([-1000, 750, 500, 2200])
        plt.tight_layout()

        plt.savefig(f"steps/side_step_{step:04d}.png")
        plt.close()

        plt.figure()
        plt.scatter(_d[0], _d[2], c="r", s=0.01, marker=".", rasterized=True)
        plt.scatter(
            _dd[0], _dd[2], c=_dd[6], s=0.01, marker=".", rasterized=True, vmin=0, vmax=1000
        )
        plt.colorbar(shrink=0.5, label="distance to boundary [microns]")
        plt.axis([-800, 700, -1400, 0])
        plt.suptitle(f"computational time [a.u]: {step}")
        plt.tight_layout()
        plt.savefig(f"steps/top_step_{step:04d}.png")
        plt.close()
