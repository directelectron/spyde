# Colormap registry — names that anyplotlib accepts directly.
# anyplotlib uses colorcet internally; these names are passed verbatim to
# Plot2D.set_colormap() / the Electron colormap selector.

COLORMAPS: dict[str, str] = {
    "gray": "gray",
    "viridis": "viridis",
    "plasma": "plasma",
    "cividis": "cividis",
    "fire": "fire",
    "hot": "hot",
    "inferno": "inferno",
    "turbo": "turbo",
    "CET-L1": "CET-L1",   # perceptually-uniform grey
    "CET-L3": "CET-L3",   # fire alias
    "CET-R1": "CET-R1",   # diverging
}

DEFAULT_COLORMAP = "gray"
