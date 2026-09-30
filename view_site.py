#%%
import numpy as np
import rasterio
import matplotlib.pyplot as plt

site = "c"    # change to "i", "c", or "e"
year = 2018   # change to any year 2013-2022

#%%
path = f"data/subsite_{site}/subsite_{site}_ndvi/subsite_{site}_{year}.tif"

with rasterio.open(path) as src:
    red = src.read(1).astype(np.float32)
    nir = src.read(2).astype(np.float32)

denom = nir + red
ndvi = np.zeros_like(denom)
valid = denom > 0
ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
biomass = np.clip(ndvi, 0, None) * 1500.0

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(biomass, cmap="YlGn")
ax.set_axis_off()
ax.set_aspect(1.0)
fig.tight_layout(pad=0)
plt.show()

# %%
transform = src.transform
extent = rasterio.plot.plotting_extent(src)

ax.imshow(biomass, cmap="YlGn", extent=extent)
ax.set_aspect('equal')
# %%
size = min(biomass.shape)
biomass_square = biomass[:size, :size]
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(biomass_square, cmap="YlGn")
ax.set_axis_off()
ax.set_aspect(1.0)
fig.tight_layout(pad=0)
plt.show()
# %%
