
import numpy as np
from scipy.spatial import cKDTree

class RegridCache:
    """
    Cache object to store precomputed KDTree, neighbor info,
    and expected counts for source and target grids.
    Includes filtering so only source pixels within the target pixel footprint are used.
    """
    def __init__(self, x_source, y_source, x_target, y_target, fraction=.9, min_per_pixel=1):
        self.x_source = x_source
        self.y_source = y_source
        self.x_target = x_target
        self.y_target = y_target
        self.fraction = fraction
        self.min_per_pixel = min_per_pixel
        # Flatten coordinates
        self.src_points = np.column_stack((x_source.ravel(), y_source.ravel()))
        self.tgt_points = np.column_stack((x_target.ravel(), y_target.ravel()))
        # Build KDTree for source points (we query *source* for neighbors)
        self.tree_src = cKDTree(self.src_points)
        # Estimate source and target pixel spacing
        dx_src = np.median(np.diff(np.unique(x_source)))
        dy_src = np.median(np.diff(np.unique(y_source)))
        dx_tgt = np.median(np.diff(np.unique(x_target)))
        dy_tgt = np.median(np.diff(np.unique(y_target)))
        # Areas
        area_src = dx_src * dy_src
        area_tgt = dx_tgt * dy_tgt
        # Expected count of source pixels in a target pixel footprint
        expected_count = (area_tgt / area_src) * self.fraction
        self.expected_count = max(min_per_pixel, int(np.ceil(expected_count)))

        # Search radius = target pixel diagonal
        search_radius = np.sqrt((dx_tgt / 2)**2 + (dy_tgt / 2)**2)
        # Precompute valid neighbor indices for each target pixel
        self.neighbor_indices = []
        for xt, yt in self.tgt_points:
            idxs = self.tree_src.query_ball_point([xt, yt], r=search_radius)
            # Filter: keep only points inside rectangular footprint
            idxs = [
                idx for idx in idxs
                if abs(self.src_points[idx, 0] - xt) <= dx_tgt / 2 and
                    abs(self.src_points[idx, 1] - yt) <= dy_tgt / 2
            ]
            print(f"Target pixel ({xt}, {yt}) has {len(idxs)} valid neighbors, expected {self.expected_count}")
            self.neighbor_indices.append(np.array(idxs, dtype=int))

def regrid_with_cache(data, cache):
    """
    Regrid a single field using a precomputed RegridCache.
    Parameters
    ----------
    data : 2D array
        Source data values.
    cache : RegridCache
        Precomputed cache for source and target grids.
    Returns
    -------
    data_target : 2D array
        Regridded data (NaN where masked).
    mask_valid : 2D boolean array
        True where target pixels have sufficient source pixels.
    """
    src_data_flat = data.ravel()
    tgt_values = np.full(cache.tgt_points.shape[0], np.nan, dtype=float)
    tgt_mask = np.zeros(cache.tgt_points.shape[0], dtype=bool)
    for i, idxs in enumerate(cache.neighbor_indices):
        # Remove NaNs from selected source pixels
        valid = idxs[np.isfinite(src_data_flat[idxs])]
        if valid.size >= cache.expected_count:
            tgt_values[i] = np.mean(src_data_flat[valid])
            tgt_mask[i] = True
    return tgt_values.reshape(cache.x_target.shape), tgt_mask.reshape(cache.x_target.shape)