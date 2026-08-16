import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# GLOBAL FIGURE SETTINGS
# =========================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

FIGURE_DPI = 600


# =========================================================
# 1. Project folders
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
BOUNDARY_DIR = DATA_DIR / "boundary"
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. Input files
# =========================================================

boundary_file = (
    BOUNDARY_DIR /
    "Maharashtra_Boundary.shp"
)

points_file = (
    DATA_DIR /
    "Maharashtra_Climate_Sampling_Points.csv"
)


# =========================================================
# 3. Check files
# =========================================================

if not boundary_file.exists():
    raise FileNotFoundError(
        f"Boundary file not found:\n{boundary_file}"
    )

if not points_file.exists():
    raise FileNotFoundError(
        f"Sampling points file not found:\n{points_file}"
    )


# =========================================================
# 4. Load Maharashtra boundary
# =========================================================

boundary = gpd.read_file(
    boundary_file
)

print("\n--- BOUNDARY ---")
print("CRS:", boundary.crs)
print("Features:", len(boundary))


# =========================================================
# 5. Load sampling points
# =========================================================

points_df = pd.read_csv(
    points_file
)

print("\n--- SAMPLING POINTS ---")
print("Rows:", len(points_df))
print(points_df.head())


# =========================================================
# 6. Convert points to GeoDataFrame
# =========================================================

points = gpd.GeoDataFrame(

    points_df,

    geometry=gpd.points_from_xy(
        points_df["longitude"],
        points_df["latitude"]
    ),

    crs="EPSG:4326"
)


# =========================================================
# 7. Match CRS with Maharashtra boundary
# =========================================================

if boundary.crs is None:

    print(
        "\nBoundary CRS was missing. "
        "Assigning EPSG:4326."
    )

    boundary = boundary.set_crs(
        "EPSG:4326"
    )


points = points.to_crs(
    boundary.crs
)


# =========================================================
# 8. Validate that points fall inside Maharashtra
# =========================================================

maharashtra_geometry = boundary.geometry.union_all()

inside = points.geometry.within(
    maharashtra_geometry
)

print("\nPoints inside Maharashtra:")
print(
    f"{inside.sum()} / {len(points)}"
)


# =========================================================
# 9. Create map
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.2, 7.2)
)


# Maharashtra polygon
boundary.plot(
    ax=ax,
    facecolor="whitesmoke",
    edgecolor="black",
    linewidth=1.2
)


# Sampling points
points.plot(
    ax=ax,
    marker="o",
    markersize=32,
    edgecolor="black",
    linewidth=0.6
)


# =========================================================
# 10. Label sampling points P01-P23
# =========================================================

for _, row in points.iterrows():

    ax.annotate(

        text=row["point_id"],

        xy=(
            row.geometry.x,
            row.geometry.y
        ),

        xytext=(4, 4),

        textcoords="offset points",

        fontsize=8,

        fontfamily="Times New Roman",

        ha="left",
        va="bottom"
    )


# =========================================================
# 11. Axis labels
# =========================================================

ax.set_xlabel(
    "Longitude (°E)"
)

ax.set_ylabel(
    "Latitude (°N)"
)


# =========================================================
# 12. Add light coordinate grid
# =========================================================

ax.grid(
    True,
    linestyle="--",
    linewidth=0.4,
    alpha=0.5
)


# =========================================================
# 13. Maintain geographical appearance
# =========================================================

ax.set_aspect(
    "equal",
    adjustable="box"
)


# =========================================================
# 14. Remove unnecessary margins
# =========================================================

minx, miny, maxx, maxy = (
    boundary.total_bounds
)

x_margin = (
    maxx - minx
) * 0.04

y_margin = (
    maxy - miny
) * 0.04


ax.set_xlim(
    minx - x_margin,
    maxx + x_margin
)

ax.set_ylim(
    miny - y_margin,
    maxy + y_margin
)


# =========================================================
# 15. Caption-style note within figure
# =========================================================

ax.text(

    0.02,
    0.02,

    "23 systematic sampling locations",

    transform=ax.transAxes,

    fontsize=9,

    fontfamily="Times New Roman",

    verticalalignment="bottom",

    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85
    )
)


# =========================================================
# 16. Layout
# =========================================================

plt.tight_layout()


# =========================================================
# 17. Save 600-DPI PNG
# =========================================================

png_file = (
    FIGURES_DIR /
    "study_area_maharashtra.png"
)

plt.savefig(
    png_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)


# =========================================================
# 18. Also save vector PDF
# =========================================================

pdf_file = (
    FIGURES_DIR /
    "study_area_maharashtra.pdf"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# 19. Confirmation
# =========================================================

print("\nStudy-area figure generated successfully.")

print("PNG:")
print(png_file)

print("\nPDF:")
print(pdf_file)

print("\nResolution: 600 DPI for PNG")
print("Font: Times New Roman")