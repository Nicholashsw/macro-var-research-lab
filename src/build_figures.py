"""
Build the diagrams that will be embedded in the macro logic PDF.
Each saved as a high-DPI PNG.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

# repo-relative; outputs/ is gitignored so a re-run never clobbers committed figures.
ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parent.parent))
FIG = ROOT / "outputs" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

NAVY = "#0B1F3A"
BLUE = "#1F4F8C"
TEAL = "#2E8B8B"
ORANGE = "#D97706"
RED = "#C44545"
GREEN = "#4A8B4A"
GREY = "#6B7280"
LIGHT_BG = "#F7F8FA"


# ============================================================
# 1. TRANSMISSION MECHANISM FLOWCHART
# ============================================================
def transmission_flow():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, color=NAVY, fc="white", fontsize=9.5, weight="normal"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              ec=color, fc=fc, lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=color, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color=GREY, style="->", lw=1.5):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=15, color=color, lw=lw))

    # Title
    ax.text(7, 8.5, "How a Fed rate hike transmits to GDP and inflation",
            ha="center", fontsize=13, weight="bold", color=NAVY)

    # Trigger
    box(5.5, 7.0, 3, 0.9, "Fed raises Fed Funds Rate", color=NAVY, fc=NAVY, fontsize=11, weight="bold")
    ax.text(7, 7.45, "Fed raises Fed Funds Rate", ha="center", va="center",
            fontsize=11, color="white", weight="bold")

    # Five channels (boxes across)
    ch_y = 5.0
    channels = [
        (0.2, "Interest rate\nchannel", BLUE),
        (3.0, "Credit\nchannel", BLUE),
        (5.8, "Asset price\n/ wealth", BLUE),
        (8.6, "Exchange rate\nchannel", BLUE),
        (11.4, "Expectations\nchannel", BLUE),
    ]
    for x, label, c in channels:
        box(x, ch_y, 2.4, 0.95, label, color=c, fc="white", fontsize=9.5, weight="bold")
        # arrow from trigger
        arrow(7, 7.0, x + 1.2, ch_y + 0.95)

    # Mechanism boxes
    mech_y = 3.0
    mechanisms = [
        (0.2, "Higher hurdle\nrate → less\nCapex, less\nconsumption"),
        (3.0, "Wider spreads,\ntighter lending,\nrisk premium\nrises"),
        (5.8, "Equities ↓,\nhousing ↓,\nwealth effect\nlowers C"),
        (8.6, "USD ↑,\nexports ↓,\nimports ↑\n→ NX ↓"),
        (11.4, "Forward path of\nrates higher,\nlong rates ↑"),
    ]
    for x, m in mechanisms:
        box(x, mech_y, 2.4, 1.35, m, color=GREY, fc=LIGHT_BG, fontsize=8.5)
        arrow(x + 1.2, ch_y, x + 1.2, mech_y + 1.35)

    # Bottom outcome
    box(2.5, 1.0, 4, 1.2, "GDP ↓\n(after 12-18 month lag)",
        color=RED, fc=RED, fontsize=11.5, weight="bold")
    ax.text(4.5, 1.6, "GDP ↓", ha="center", va="center",
            fontsize=14, color="white", weight="bold")
    ax.text(4.5, 1.25, "(peak at 12-18 month lag)", ha="center", va="center",
            fontsize=9, color="white")

    box(7.5, 1.0, 4, 1.2, "Inflation ↓\n(after 18-24 month lag)",
        color=GREEN, fc=GREEN, fontsize=11.5, weight="bold")
    ax.text(9.5, 1.6, "Inflation ↓", ha="center", va="center",
            fontsize=14, color="white", weight="bold")
    ax.text(9.5, 1.25, "(peak at 18-24 month lag)", ha="center", va="center",
            fontsize=9, color="white")

    # Arrows from mechanisms to outcomes
    for x, _ in mechanisms:
        arrow(x + 1.2, mech_y, 4.5 if x < 7 else 9.5, 2.2, color=GREY, lw=1.0)

    plt.savefig(FIG / "01_transmission_flow.png", facecolor="white")
    plt.close()


# ============================================================
# 2. PHILLIPS-CURVE / TAYLOR RULE ILLUSTRATION
# ============================================================
def taylor_phillips():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Phillips curve
    pi_e = 2.0
    u_star = 4.5
    u = np.linspace(2.5, 8, 200)
    pi = pi_e + 1.5 * (u_star - u)  # short-run Phillips, kappa=1.5

    ax1.plot(u, pi, color=NAVY, lw=2.5, label=r"$\pi_t = \pi_t^e - \kappa(u_t - u^*)$")
    ax1.axhline(pi_e, color=GREY, ls=":", lw=1, alpha=0.7)
    ax1.axvline(u_star, color=GREY, ls=":", lw=1, alpha=0.7)
    ax1.scatter([u_star], [pi_e], color=RED, zorder=5, s=80, label="long-run equilibrium")
    ax1.annotate(r"$u^*$ (NAIRU)", xy=(u_star, 0), xytext=(u_star + 0.3, -0.8),
                 fontsize=10, color=GREY)
    ax1.annotate(r"$\pi^e$", xy=(2.5, pi_e), xytext=(2.6, pi_e + 0.3),
                 fontsize=11, color=GREY)
    ax1.set_xlabel("Unemployment rate $u_t$ (%)")
    ax1.set_ylabel(r"Inflation $\pi_t$ (%)")
    ax1.set_title("Short-run Phillips curve\n(higher u → lower inflation)",
                  fontsize=11, weight="bold", color=NAVY)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-2, 8)

    # Taylor rule
    r_star = 0.5  # neutral real rate
    pi_target = 2.0
    pi_arr = np.linspace(0, 6, 200)
    # Taylor rule: i = r* + π + 0.5(π-π*) + 0.5(y-y*)/y*
    # Assume output gap = 0 for now
    i_taylor = r_star + pi_arr + 0.5 * (pi_arr - pi_target)

    ax2.plot(pi_arr, i_taylor, color=NAVY, lw=2.5,
             label=r"$i_t = r^* + \pi_t + 0.5(\pi_t - \pi^*) + 0.5\tilde{y}_t$")
    ax2.plot(pi_arr, pi_arr, color=GREY, lw=1, ls="--",
             label=r"$i = \pi$ (Fisher line)")
    ax2.axhline(r_star + pi_target, color=RED, ls=":", lw=1, alpha=0.7)
    ax2.axvline(pi_target, color=RED, ls=":", lw=1, alpha=0.7)
    ax2.scatter([pi_target], [r_star + pi_target + 0.5*0], color=RED, zorder=5, s=80,
                label="neutral stance")
    ax2.fill_between(pi_arr, i_taylor, pi_arr, where=(i_taylor > pi_arr),
                     color=ORANGE, alpha=0.15, label="real rate above zero (contractionary)")
    ax2.set_xlabel(r"Inflation $\pi_t$ (%)")
    ax2.set_ylabel(r"Policy rate $i_t$ (%)")
    ax2.set_title("Taylor rule\n(higher inflation → bigger rate response)",
                  fontsize=11, weight="bold", color=NAVY)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(FIG / "02_taylor_phillips.png", facecolor="white")
    plt.close()


# ============================================================
# 3. IMPULSE RESPONSE: GDP AND INFLATION TO RATE SHOCK
# ============================================================
def irf_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    h = np.arange(0, 21)

    # Stylised IRFs based on Gertler-Karadi-style magnitudes
    # GDP: hump-shaped negative, peak ~quarter 5-8
    gdp_irf = -0.45 * np.exp(-((h - 6) ** 2) / 25) * (1 - np.exp(-h/2))
    gdp_lo = gdp_irf - 0.15
    gdp_hi = gdp_irf + 0.15

    # Inflation: slower, peaks later, less front-loaded
    inf_irf = -0.30 * np.exp(-((h - 10) ** 2) / 40) * (1 - np.exp(-h/3))
    inf_lo = inf_irf - 0.12
    inf_hi = inf_irf + 0.12

    ax1.plot(h, gdp_irf, color=NAVY, lw=2.5)
    ax1.fill_between(h, gdp_lo, gdp_hi, color=NAVY, alpha=0.2, label="68% CI")
    ax1.axhline(0, color="black", lw=1)
    ax1.axvline(6, color=RED, ls=":", lw=1.2, alpha=0.7)
    ax1.annotate("peak impact ≈ Q6\n(~18 months)", xy=(6, gdp_irf[6]),
                 xytext=(10, -0.55), fontsize=9, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax1.set_xlabel("Quarters after 25bp rate shock")
    ax1.set_ylabel("Log GDP response (pp)")
    ax1.set_title("IRF: GDP response to monetary tightening",
                  fontsize=11, weight="bold", color=NAVY)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-0.75, 0.25)

    ax2.plot(h, inf_irf, color=NAVY, lw=2.5)
    ax2.fill_between(h, inf_lo, inf_hi, color=NAVY, alpha=0.2, label="68% CI")
    ax2.axhline(0, color="black", lw=1)
    ax2.axvline(10, color=RED, ls=":", lw=1.2, alpha=0.7)
    ax2.annotate("peak impact ≈ Q10\n(~2.5 years)", xy=(10, inf_irf[10]),
                 xytext=(13, -0.40), fontsize=9, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax2.set_xlabel("Quarters after 25bp rate shock")
    ax2.set_ylabel("Inflation response (pp)")
    ax2.set_title("IRF: Inflation response to monetary tightening",
                  fontsize=11, weight="bold", color=NAVY)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.55, 0.20)

    plt.tight_layout()
    plt.savefig(FIG / "03_irfs.png", facecolor="white")
    plt.close()


# ============================================================
# 4. YIELD CURVE - DIFFERENT SHAPES
# ============================================================
def yield_curves():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    maturity = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

    normal     = np.array([2.0, 2.2, 2.5, 2.9, 3.2, 3.7, 4.0, 4.3, 4.6, 4.7])
    flat       = np.array([4.5, 4.55, 4.6, 4.65, 4.7, 4.7, 4.7, 4.7, 4.75, 4.75])
    inverted   = np.array([5.4, 5.3, 5.1, 4.7, 4.4, 4.1, 4.0, 4.0, 4.1, 4.2])
    steepening = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 2.7, 3.2, 3.9, 4.6, 4.9])

    ax.plot(maturity, normal,     "o-", color=GREEN, lw=2.2, label="Normal (expansion)")
    ax.plot(maturity, flat,       "s-", color=ORANGE, lw=2.2, label="Flat (late cycle)")
    ax.plot(maturity, inverted,   "^-", color=RED, lw=2.2, label="Inverted (recession signal)")
    ax.plot(maturity, steepening, "d-", color=BLUE, lw=2.2, label="Bull steepening (easing cycle)")

    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 2, 5, 10, 30])
    ax.set_xticklabels(["3M", "1Y", "2Y", "5Y", "10Y", "30Y"])
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Four canonical yield curve shapes",
                 fontsize=12, weight="bold", color=NAVY)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.axhline(0, color="black", lw=0.5)

    # Add annotation
    ax.text(0.27, 5.7, "Inverted 2y-10y has preceded\nevery US recession since 1955",
            fontsize=9, color=RED, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT_BG, ec=RED, alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG / "04_yield_curves.png", facecolor="white")
    plt.close()


# ============================================================
# 5. UIP / CARRY DIAGRAM
# ============================================================
def uip_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Why higher US rates strengthen the USD: UIP and capital flows",
                 fontsize=12, weight="bold", color=NAVY, pad=15)

    def box(x, y, w, h, text, color=NAVY, fc="white", fontsize=9.5, weight="normal", txt_color=None):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              ec=color, fc=fc, lw=1.8)
        ax.add_patch(rect)
        c = txt_color if txt_color else color
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=c, weight=weight)

    def arrow(x1, y1, x2, y2, color=GREY, label=None, lw=1.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                     mutation_scale=18, color=color, lw=lw))
        if label:
            ax.text((x1+x2)/2 + 0.15, (y1+y2)/2 + 0.2, label, fontsize=8.5,
                    color=color, style="italic")

    # Top: i_US rises
    box(4.5, 5.5, 3, 1.0, "$i_{US} \\uparrow$  (Fed hikes)",
        color=NAVY, fc=NAVY, fontsize=11, weight="bold", txt_color="white")

    # Two paths
    box(0.5, 3.0, 3.5, 1.3, "Higher USD returns\nattract foreign capital",
        color=BLUE, fontsize=10)
    box(8.0, 3.0, 3.5, 1.3, "Domestic investors\nsell foreign assets,\nrepatriate to USD",
        color=BLUE, fontsize=10)

    arrow(5.5, 5.5, 2.25, 4.3)
    arrow(6.5, 5.5, 9.75, 4.3)

    # Capital flows
    box(3.5, 1.0, 5.0, 1.3, "Capital inflows to USD",
        color=TEAL, fc=TEAL, fontsize=11, weight="bold", txt_color="white")
    arrow(2.25, 3.0, 4.5, 2.3)
    arrow(9.75, 3.0, 7.5, 2.3)

    # UIP equation
    ax.text(6, 0.3,
            r"UIP condition:   $i_{US} - i_{foreign} \approx E[\Delta \log S]$",
            ha="center", fontsize=10.5, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT_BG, ec=NAVY))

    plt.savefig(FIG / "05_uip_flows.png", facecolor="white")
    plt.close()


# ============================================================
# 6. SIGN MAP - HOW EACH VARIABLE RESPONDS
# ============================================================
def sign_map():
    fig, ax = plt.subplots(figsize=(11, 6))

    variables = ["Real GDP", "Inflation", "Unemployment", "Equity prices",
                 "Housing", "Investment", "USD (FX)", "Imports", "Net exports",
                 "Credit spreads", "Long yields", "Bond prices"]

    short_run = [-1, 0, +1, -1, -1, -1, +1, +1, -1, +1, +1, -1]   # 0-12 months
    medium_run = [-1, -1, +1, -1, -1, -1, +1, 0, -1, +1, +1, -1]  # 12-24 months
    long_run = [0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]             # 24+ months

    cmap_vals = np.array([short_run, medium_run, long_run]).T

    im = ax.imshow(cmap_vals, cmap="RdYlGn_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Short run\n(0-12M)", "Medium run\n(12-24M)", "Long run\n(24M+)"])

    # Annotate cells
    for i in range(len(variables)):
        for j in range(3):
            v = cmap_vals[i, j]
            label = "↓" if v == -1 else ("↑" if v == +1 else "~")
            color = "white" if abs(v) == 1 else "black"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=18, color=color, weight="bold")

    ax.set_title("Sign of response to a Fed rate hike",
                 fontsize=12, weight="bold", color=NAVY, pad=15)

    # Custom legend
    legend = [
        mpatches.Patch(color="#C44545", label="↑  variable rises"),
        mpatches.Patch(color="#4A8B4A", label="↓  variable falls"),
        mpatches.Patch(color="#FFF59D", label="~  little / no effect"),
    ]
    ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(FIG / "06_sign_map.png", facecolor="white")
    plt.close()


# ============================================================
# 7. SINGAPORE VS US POLICY FRAMEWORK
# ============================================================
def sg_vs_us():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("US Fed vs Singapore MAS: different policy instruments, same goal",
                 fontsize=12, weight="bold", color=NAVY, pad=10)

    def box(x, y, w, h, text, color=NAVY, fc="white", fontsize=9.5, weight="normal", txt_color=None):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              ec=color, fc=fc, lw=1.8)
        ax.add_patch(rect)
        c = txt_color if txt_color else color
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=c, weight=weight)

    # Headers
    box(0.5, 6.8, 6, 1.0, "US Federal Reserve", color=BLUE, fc=BLUE,
        fontsize=13, weight="bold", txt_color="white")
    box(7.5, 6.8, 6, 1.0, "Singapore (MAS)", color=ORANGE, fc=ORANGE,
        fontsize=13, weight="bold", txt_color="white")

    # Instrument row
    ax.text(0.2, 5.6, "Instrument:", fontsize=10, weight="bold", color=NAVY)
    box(0.5, 4.7, 6, 0.8, "Fed Funds Rate (price of money)", color=BLUE, fontsize=10)
    ax.text(7.2, 5.6, "Instrument:", fontsize=10, weight="bold", color=NAVY)
    box(7.5, 4.7, 6, 0.8, "SGD NEER slope/level/width", color=ORANGE, fontsize=10)

    # Channel
    ax.text(0.2, 3.6, "Channel:", fontsize=10, weight="bold", color=NAVY)
    box(0.5, 2.7, 6, 0.8, "Interest rate / credit / wealth", color=BLUE, fontsize=10)
    ax.text(7.2, 3.6, "Channel:", fontsize=10, weight="bold", color=NAVY)
    box(7.5, 2.7, 6, 0.8, "Exchange rate (import prices)", color=ORANGE, fontsize=10)

    # Why
    ax.text(0.2, 1.6, "Why this works:", fontsize=10, weight="bold", color=NAVY)
    box(0.5, 0.4, 6, 1.0, "Large, closed-ish economy.\nDomestic demand drives inflation.",
        color=BLUE, fontsize=9)
    ax.text(7.2, 1.6, "Why this works:", fontsize=10, weight="bold", color=NAVY)
    box(7.5, 0.4, 6, 1.0, "Small, open economy. Exports ~170% of GDP.\nImported inflation matters most.",
        color=ORANGE, fontsize=9)

    plt.savefig(FIG / "07_sg_vs_us.png", facecolor="white")
    plt.close()


# ============================================================
# 8. THE COEFFICIENT SIGNS - WHAT YOU ASKED FOR
# ============================================================
def coefficient_signs():
    """Visualise the sign of regression coefficients in a VAR / Phillips / IS curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: IS curve ----
    ax = axes[0]
    r = np.linspace(0, 8, 100)
    # GDP gap = a - b * r  (negative slope)
    gdp_gap = 3.0 - 0.6 * r
    ax.plot(r, gdp_gap, color=NAVY, lw=2.5, label=r"$\tilde y_t = \alpha - \beta r_t$")
    ax.axhline(0, color="black", lw=0.7)
    ax.fill_between(r, gdp_gap, 0, where=(gdp_gap > 0), color=GREEN, alpha=0.15)
    ax.fill_between(r, gdp_gap, 0, where=(gdp_gap < 0), color=RED, alpha=0.15)
    ax.set_xlabel("Real interest rate $r_t$ (%)")
    ax.set_ylabel("Output gap $\\tilde y_t$ (% from potential)")
    ax.set_title("IS curve: $\\beta > 0$\nhigher real rates $\\to$ lower GDP",
                 fontsize=11, weight="bold", color=NAVY)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(6, 1.5, "boom\n($\\tilde y > 0$)", fontsize=9, color=GREEN, ha="center")
    ax.text(1, -1.5, "slack\n($\\tilde y < 0$)", fontsize=9, color=RED, ha="center")

    # ---- Right: Phillips curve table ----
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Sign of key macro coefficients\n(canonical New Keynesian model)",
                 fontsize=11, weight="bold", color=NAVY)

    # Table
    rows = [
        ("Equation", "Coefficient", "Sign", "Reads as"),
        (r"$\tilde y_t = \alpha - \beta r_t + \epsilon_t^{IS}$",
         r"$\beta$ on $r_t$", "$+$ (so $-\beta$ on $r$)", "Higher rate $\\to$ lower GDP"),
        (r"$\pi_t = \pi^e_t + \kappa \tilde y_t + \epsilon_t^{PC}$",
         r"$\kappa$ on $\tilde y$", "$+$", "Tighter slack $\\to$ higher inflation"),
        (r"$i_t = r^* + \pi_t + \phi_\pi(\pi_t-\pi^*) + \phi_y \tilde y_t$",
         r"$\phi_\pi$", "$+$ ($>1$, Taylor principle)", "Above-target $\\pi$ $\\to$ raise $i$ more"),
        (r"NX$_t = \gamma_1 y^*_t - \gamma_2 q_t$",
         r"$\gamma_2$ on REER $q$", r"$+$ (so $-\gamma_2$ on $q$)", "Stronger FX $\\to$ lower exports"),
        ("UIP:  $i_{US} - i^* = E_t[\\Delta s_{t+1}]$",
         r"slope of $\Delta s$", "$+$ on $i_{US}$", "Higher US rate $\\to$ expected USD depreciation"),
    ]

    n = len(rows)
    col_x = [0.0, 0.32, 0.55, 0.72]
    col_w = [0.32, 0.23, 0.17, 0.28]
    for ri, row in enumerate(rows):
        y = 0.92 - ri * 0.15
        for ci, txt in enumerate(row):
            fc = LIGHT_BG if ri == 0 else "white"
            fontsize = 9 if ri > 0 else 9.5
            weight = "bold" if ri == 0 else "normal"
            ax.text(col_x[ci] + 0.01, y, txt, fontsize=fontsize, weight=weight,
                    color=NAVY, va="center")
        if ri == 0:
            ax.axhline(y - 0.06, color=NAVY, lw=1, xmin=0, xmax=1)

    plt.tight_layout()
    plt.savefig(FIG / "08_coefficient_signs.png", facecolor="white")
    plt.close()


# Build all
for fn in [transmission_flow, taylor_phillips, irf_chart, yield_curves,
           uip_diagram, sign_map, sg_vs_us, coefficient_signs]:
    fn()
    print(f"OK: {fn.__name__}")

print("\nAll figures in", FIG)
