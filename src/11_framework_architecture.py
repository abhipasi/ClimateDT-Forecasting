import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path


# =========================================================
# Global settings
# =========================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

FIGURE_DPI = 600


# =========================================================
# Project folders
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


# =========================================================
# Helper functions
# =========================================================

def add_box(ax, x, y, w, h, text, fontsize=11):
    rect = Rectangle(
        (x, y), w, h,
        fill=False,
        linewidth=1.2
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.2
    )
    ax.add_patch(arrow)


# =========================================================
# Create figure
# =========================================================

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")


# =========================================================
# Top source boxes
# =========================================================

add_box(ax, 1.0, 10.5, 2.6, 0.9, "CHIRPS v3 Daily\nRainfall")
add_box(ax, 6.4, 10.5, 2.6, 0.9, "ERA5-Land Daily\nTemperature")

add_arrow(ax, 2.3, 10.5, 4.8, 9.8)
add_arrow(ax, 7.7, 10.5, 5.2, 9.8)


# =========================================================
# Main vertical flow
# =========================================================

add_box(ax, 3.3, 9.0, 3.4, 0.9, "Google Earth Engine\nData Extraction")
add_box(ax, 3.3, 7.8, 3.4, 0.9, "Spatial and Temporal\nHarmonisation")
add_box(ax, 3.3, 6.5, 3.4, 0.9, "Feature Engineering\n(Lags, Rolling Means,\nSeasonal Encoding)")
add_box(ax, 3.3, 5.1, 3.4, 0.9, "Forecasting Models")

add_arrow(ax, 5.0, 9.0, 5.0, 8.7)
add_arrow(ax, 5.0, 7.8, 5.0, 7.4)
add_arrow(ax, 5.0, 6.5, 5.0, 6.0)


# =========================================================
# Model boxes
# =========================================================

add_box(ax, 0.8, 3.7, 1.8, 0.9, "Persistence")
add_box(ax, 3.0, 3.7, 1.8, 0.9, "Random\nForest")
add_box(ax, 5.2, 3.7, 1.8, 0.9, "XGBoost")
add_box(ax, 7.4, 3.7, 1.8, 0.9, "LSTM")

add_arrow(ax, 5.0, 5.1, 1.7, 4.6)
add_arrow(ax, 5.0, 5.1, 3.9, 4.6)
add_arrow(ax, 5.0, 5.1, 6.1, 4.6)
add_arrow(ax, 5.0, 5.1, 8.3, 4.6)


# =========================================================
# Merge into forecasting output
# =========================================================

add_box(ax, 3.0, 2.1, 4.0, 0.9,
        "One-Day-Ahead Rainfall and\nTemperature Forecasting")

add_arrow(ax, 1.7, 3.7, 4.0, 3.0)
add_arrow(ax, 3.9, 3.7, 4.6, 3.0)
add_arrow(ax, 6.1, 3.7, 5.4, 3.0)
add_arrow(ax, 8.3, 3.7, 6.0, 3.0)


# =========================================================
# Final output boxes
# =========================================================

add_box(ax, 3.0, 0.9, 4.0, 0.8,
        "Forecast Climate-State Representation")

add_box(ax, 3.0, 0.0, 4.0, 0.8,
        "Toward a Climate Digital Twin")

add_arrow(ax, 5.0, 2.1, 5.0, 1.7)
add_arrow(ax, 5.0, 0.9, 5.0, 0.8)


# =========================================================
# Save figure
# =========================================================

png_file = FIGURES_DIR / "framework_architecture.png"
pdf_file = FIGURES_DIR / "framework_architecture.pdf"

plt.tight_layout()

plt.savefig(
    png_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.close()

print("Framework architecture figure generated successfully.")
print("PNG:", png_file)
print("PDF:", pdf_file)
print("Resolution: 600 DPI")
print("Font: Times New Roman")