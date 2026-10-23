"""Robinson maps with aligned longitude and significance arrays."""
from __future__ import annotations

import numpy as np


def projection():
    import cartopy.crs as ccrs
    return ccrs.Robinson(central_longitude=0)


def map_panel(ax, lon, lat, values, significant=None, vmin=None, vmax=None,
              cmap='RdBu_r', title='', cbar_label='', mask=None, norm=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lon, lat = np.asarray(lon).ravel(), np.asarray(lat).ravel()
    values = np.asarray(values, dtype=float)
    if values.shape != (len(lat), len(lon)):
        raise ValueError(f'Map data must be latitude x longitude, found {values.shape}.')
    wrapped = (lon + 180) % 360 - 180
    order = np.argsort(wrapped)
    wrapped, values = wrapped[order], values[:, order]
    if len(np.unique(wrapped)) != len(wrapped):
        raise ValueError('Longitude contains a duplicated cyclic endpoint.')
    hidden = ~np.isfinite(values)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != values.shape:
            raise ValueError('Mask shape differs from the map.')
        excluded = np.isfinite(mask[:, order]) & (mask[:, order] != 0)
        hidden |= excluded
    plate = ccrs.PlateCarree()
    y_limit = abs(ax.projection.transform_point(0, 60, plate)[1])
    ax.set_xlim(*ax.projection.x_limits)
    ax.set_ylim(-y_limit, y_limit)
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='0.88', edgecolor='none', zorder=0)
    kwargs = {'cmap': cmap, 'transform': plate, 'shading': 'auto', 'rasterized': True}
    if norm is not None:
        kwargs['norm'] = norm
    else:
        kwargs.update(vmin=vmin, vmax=vmax)
    mappable = ax.pcolormesh(wrapped, lat, np.ma.array(values, mask=hidden), **kwargs)
    if significant is not None:
        sig = np.asarray(significant)
        if sig.shape != values.shape:
            raise ValueError('Significance shape differs from the map.')
        sig = np.where((sig[:, order] == 1) & ~hidden, 1.0, np.nan)
        if np.isfinite(sig).any():
            ax.contourf(wrapped, lat, sig, levels=[0.5, 1.5], colors='none',
                        hatches=['////'], transform=plate)
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.set_title(title)
    if cbar_label:
        colorbar = ax.figure.colorbar(mappable, ax=ax, orientation='horizontal', fraction=0.055, pad=0.08)
        colorbar.set_label(cbar_label)
    return mappable
