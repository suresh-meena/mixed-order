def apply_pub_style():
    """Set global rcParams for a clean, thesis-ready look (Palatino + Euler VM)."""
    import shutil, subprocess
    import matplotlib.pyplot as plt

    def _pkg_available(sty: str) -> bool:
        """Return True if a LaTeX .sty file is locatable via kpsewhich."""
        if shutil.which("kpsewhich") is None:
            return False
        r = subprocess.run(["kpsewhich", sty],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())

    have_latex    = shutil.which("latex") is not None
    have_dvipng   = shutil.which("dvipng") is not None
    have_mathpazo = _pkg_available("mathpazo.sty")
    have_eulervm  = _pkg_available("eulervm.sty")
    have_type1cm  = _pkg_available("type1cm.sty")

    if have_latex and have_dvipng and have_type1cm and (have_mathpazo or have_eulervm):
        preamble = ""
        if have_mathpazo:
            preamble += r"\usepackage{mathpazo}"   # Palatino text + Pazo math
        if have_eulervm:
            preamble += r"\usepackage{eulervm}"    # Euler VM overlaid for math
        if not have_eulervm:
            print("  [style] eulervm not found — using mathpazo only.")
        plt.rcParams["text.usetex"]         = True
        plt.rcParams["text.latex.preamble"] = preamble
    elif have_mathpazo or have_eulervm:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.fontset"] = "cm"
        print("  [style] LaTeX packages found but TeX prerequisites are incomplete.\n"
              "  Falling back to mathtext.")
    else:
        plt.rcParams["mathtext.fontset"] = "cm"
        print("  [style] LaTeX/mathpazo unavailable — falling back to mathtext.")

    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          13,
        "axes.titlesize":     14,
        "axes.labelsize":     13,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "figure.titlesize":   15,
        "lines.linewidth":    2.0,
        "lines.markersize":   6,
        "axes.linewidth":     1.0,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "xtick.minor.width":  0.6,
        "ytick.minor.width":  0.6,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid":          True,
        "grid.alpha":         0.35,
        "grid.linewidth":     0.6,
        "grid.linestyle":     "--",
        "figure.dpi":         150,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
    })

def get_color_palette():
    """Return the standard color dictionary used for thesis experiments."""
    return {
        'theory':     '#222222', # Charcoal/Black
        'uncentered': '#EE3333', # Red (Drift/Failure)
        'centered':   '#0077BB', # Blue (Correction)
        'mixed':      '#009988', # Teal/Green (Mixed-order highlight)
        'extra':      '#EE7733', # Orange
        'base':       '#BBBBBB', # Grey
    }
