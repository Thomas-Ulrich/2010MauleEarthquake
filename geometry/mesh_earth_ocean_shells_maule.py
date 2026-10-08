import contourpy
import gmsh
import numpy as np
import shapely
import xarray as xr
from collections import Counter
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, box
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
import scipy.ndimage as ndimage

PROJ_STR = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=286.0 +lat_0=-37.00"
M_PER_DEG = 111_000.0
SIDES = ["west", "east", "south", "north"]


def preprocess_bathymetry(ds_sub, var_name="elevation", z_zero=None,
                          smooth_range=None, smooth_sigma_m=None):
    """Smooth near the coast, then move z ~ 0 points off zero.

    :param z_zero: new elevation for points with |z| < 0.01 (e.g. -20.0).
    :param smooth_range: smooth only where |z| < smooth_range (m).
    :param smooth_sigma_m: Gaussian sigma in metres (converted to grid cells,
        separately for lat and lon, so the smoothing is isotropic on the ground).
    """
    z = ds_sub[var_name].values.astype(float).copy()
    lons = ds_sub["lon"].values
    lats = ds_sub["lat"].values

    if smooth_range and smooth_sigma_m:
        dy = abs(lats[1] - lats[0]) * 111_000.0
        dx = abs(lons[1] - lons[0]) * 111_000.0 * np.cos(np.deg2rad(lats.mean()))
        sig = (smooth_sigma_m / dy, smooth_sigma_m / dx)   # (lat axis, lon axis)
        z_smooth = ndimage.gaussian_filter(z, sigma=sig, mode="nearest")
        ids = np.abs(z) < smooth_range
        z[ids] = z_smooth[ids]
        print(f"smoothed {ids.sum()} points with |z| < {smooth_range} m "
              f"(sigma = {smooth_sigma_m} m = {sig[0]:.2f} x {sig[1]:.2f} cells)")

    if z_zero is not None:
        ids = np.abs(z) < 0.01
        z[ids] = z_zero
        print(f"{ids.sum()} points with z ~ 0 changed to z = {z_zero}")

    return ds_sub.assign({var_name: (ds_sub[var_name].dims, z)})

def extract_shorelines(ds_sub, var_name="elevation"):
    lons = ds_sub["lon"].values
    lats = ds_sub["lat"].values
    z = ds_sub[var_name].values
    cg = contourpy.contour_generator(lons, lats, z)
    return [l for l in cg.lines(0.0) if len(l) >= 2]


def build_faces(shorelines, extent, interp, simplify_deg=0.0,
                boundary_seg_deg=None, prec=1e-9):
    lon_min, lon_max, lat_min, lat_max = extent
    ring = box(lon_min, lat_min, lon_max, lat_max).exterior
    if boundary_seg_deg:
        ring = shapely.segmentize(ring, boundary_seg_deg)
    boundary = shapely.set_precision(ring, prec)

    lines = []
    for l in shorelines:
        ls = LineString(l)
        if simplify_deg > 0:
            ls = ls.simplify(simplify_deg, preserve_topology=True)
        if ls.length > 0:
            lines.append(shapely.set_precision(ls, prec))

    noded = unary_union([boundary] + lines)
    faces = []
    for poly in polygonize(noded):
        p = poly.representative_point()
        z = float(interp([[p.y, p.x]])[0])
        faces.append((orient(poly, 1.0), "land" if z > 0 else "ocean"))
    return faces


def _ekey(p, q, nd=4):
    a = (round(p[0], nd), round(p[1], nd))
    b = (round(q[0], nd), round(q[1], nd))
    return (a, b) if a <= b else (b, a)


def count_bad_edges(tri_list):
    """Edges not shared by exactly 2 triangles (0 = watertight)."""
    t = np.concatenate(tri_list)
    E = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    return int(np.sum(cnt != 2))


def _nodes_and_tris(surface_tags):
    """Return sorted node tags, coords (n,3) and, per surface, triangle indices."""
    tags, co, _ = gmsh.model.mesh.getNodes()
    co = co.reshape(-1, 3)
    order = np.argsort(tags)
    tags, co = tags[order], co[order]
    tris = []
    for s in surface_tags:
        t = gmsh.model.mesh.getElementsByType(2, s)[1].reshape(-1, 3)
        tris.append(np.searchsorted(tags, t))
    return tags, co, tris


def mesh_earth_sides(ring_xy, ring_z, ring_side, earth_depth, lc_deep, growth=1.15):
    """Mesh the vertical sides of the earth block with Gmsh.

    Each side is unrolled to a plane (s = arc length along the boundary, z).
    Top edge  : topography, exactly one element per boundary edge of the surface mesh.
    Bottom    : z = -earth_depth, spacing lc_deep.
    Corners   : shared vertical curves with a deterministic geometric distribution.

    Returns: wall_nodes (n,3), [(side_index, triangles)], bottom ring xy (m,2).
    """
    K = len(ring_xy)
    D = earth_depth
    xr = np.vstack([ring_xy, ring_xy[:1]])
    zr = np.append(ring_z, ring_z[0])
    seg = np.hypot(*np.diff(xr, axis=0).T)
    S = np.concatenate([[0.0], np.cumsum(seg)])

    bp = [0] + [i for i in range(1, K) if ring_side[i] != ring_side[i - 1]] + [K]
    W = len(bp) - 1                                  # number of sides (normally 4)
    h = np.array([0.5 * (seg[(i - 1) % K] + seg[i % K]) for i in range(K + 1)])

    gmsh.model.add("earth_sides")
    geo = gmsh.model.geo

    # top polyline (global, one point per boundary node, s = K is the copy of s = 0)
    T = [geo.addPoint(S[i], zr[i], 0.0, h[i]) for i in range(K + 1)]
    top = [geo.addLine(T[i], T[i + 1]) for i in range(K)]
    for l in top:
        geo.mesh.setTransfiniteCurve(l, 2)

    # corner vertical curves: top -> bottom, same distribution on both copies of corner 0
    Bc, V = [], []
    for c in range(W + 1):
        i = bp[c]
        Bc.append(geo.addPoint(S[i], -D, 0.0, lc_deep))
        V.append(geo.addLine(T[i], Bc[c]))
        H = zr[i] + D
        n = max(int(np.ceil(np.log(1 + H * (growth - 1) / h[i]) / np.log(growth))), 2)
        geo.mesh.setTransfiniteCurve(V[c], n + 1, "Progression", growth)

    surfs, side_idx, bottom_s = [], [], []
    for k in range(W):
        i0, i1 = bp[k], bp[k + 1]
        nb = max(1, int(round((S[i1] - S[i0]) / lc_deep)))
        sb = np.linspace(S[i0], S[i1], nb + 1)
        bpts = [Bc[k]] + [geo.addPoint(s, -D, 0.0, lc_deep) for s in sb[1:-1]] + [Bc[k + 1]]
        blines = [geo.addLine(bpts[j], bpts[j + 1]) for j in range(nb)]
        for l in blines:
            geo.mesh.setTransfiniteCurve(l, 2)
        bottom_s.extend(sb[:-1])

        loop = (top[i0:i1] + [V[k + 1]] + [-l for l in reversed(blines)] + [-V[k]])
        cl = geo.addCurveLoop(loop)
        surfs.append(geo.addPlaneSurface([cl]))
        side_idx.append(int(ring_side[i0]))
    geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_deep)
    gmsh.model.mesh.generate(2)

    tags, co, tris = _nodes_and_tris(surfs)
    s, z = co[:, 0], co[:, 1]
    wall_nodes = np.column_stack([np.interp(s, S, xr[:, 0]),
                                  np.interp(s, S, xr[:, 1]), z])

    out = []
    for t, sd in zip(tris, side_idx):
        a, b, c = co[t[:, 0], :2], co[t[:, 1], :2], co[t[:, 2], :2]
        area2 = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        t = t.copy()
        t[area2 < 0] = t[area2 < 0][:, ::-1]      # CCW in (s, z) -> outward normal
        out.append((sd, t))

    bottom_s = np.array(bottom_s)
    ring_b = np.column_stack([np.interp(bottom_s, S, xr[:, 0]),
                              np.interp(bottom_s, S, xr[:, 1])])
    gmsh.clear()
    return wall_nodes, out, ring_b


def mesh_bottom(ring_b, earth_depth, lc_deep):
    """Mesh the flat bottom; boundary nodes = bottom nodes of the sides (1 element/edge)."""
    gmsh.model.add("earth_bottom")
    geo = gmsh.model.geo
    n = len(ring_b)
    p = [geo.addPoint(x, y, -earth_depth, lc_deep) for x, y in ring_b]
    ls = [geo.addLine(p[i], p[(i + 1) % n]) for i in range(n)]
    for l in ls:
        geo.mesh.setTransfiniteCurve(l, 2)
    s = geo.addPlaneSurface([geo.addCurveLoop(ls)])
    geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_deep)
    gmsh.model.mesh.generate(2)

    tags, co, (tri,) = _nodes_and_tris([s])
    a, b, c = co[tri[:, 0], :2], co[tri[:, 1], :2], co[tri[:, 2], :2]
    area2 = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    tri = tri.copy()
    tri[area2 > 0] = tri[area2 > 0][:, ::-1]      # CW in (x, y) -> normal -z
    gmsh.clear()
    return co, tri


def generate_earth_and_sea_shells(
    netcdf_path,
    bbox,
    var_name="elevation",
    z_zero=-20.0,           # None to disable
    smooth_range=100.0,     # metres, None to disable
    smooth_sigma_m=1000.0,  # metres
    proj_str=PROJ_STR,
    coarsen_factor=1,
    lc_coarse=8000.0,       # metres
    lc_fine=500.0,          # metres
    dist_min=1000.0,
    dist_max=20000.0,
    simplify_m=None,
    earth_depth=400_000.0,  # metres
    lc_deep=50_000.0,       # element size at the bottom (sides + bottom)
    wall_growth=1.15,       # vertical growth ratio along the corner edges
    output_mesh="earth_sea_shells.msh",
    gui=False,
):
    # ---------------- 1. Data ----------------
    ds = xr.open_dataset(netcdf_path)
    lon_min, lon_max, lat_min, lat_max = bbox
    ds_sub = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    if coarsen_factor > 1:
        ds_sub = ds_sub.coarsen(
            lon=coarsen_factor, lat=coarsen_factor, boundary="trim"
        ).mean()

    ds_sub = preprocess_bathymetry(ds_sub, var_name, z_zero,
                                   smooth_range, smooth_sigma_m)


    lons = ds_sub["lon"].values
    lats = ds_sub["lat"].values
    z_grid = ds_sub[var_name].values
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    interp = RegularGridInterpolator(
        (lats, lons), z_grid, bounds_error=False, fill_value=None
    )

    fwd = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    inv_tr = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)

    # ---------------- 2. Faces (lon/lat) -> project ----------------
    if simplify_m is None:
        simplify_m = 0.25 * lc_fine
    shorelines = extract_shorelines(ds_sub, var_name)
    faces_ll = build_faces(
        shorelines, extent, interp,
        simplify_deg=simplify_m / M_PER_DEG,
        boundary_seg_deg=lc_coarse / M_PER_DEG,
    )

    def to_xy(geom):
        return shapely.transform(
            geom, lambda c: np.column_stack(fwd.transform(c[:, 0], c[:, 1]))
        )

    faces = [(orient(to_xy(p), 1.0), c) for p, c in faces_ll]

    edge_use = Counter()
    for poly, _ in faces:
        for ring in [poly.exterior, *poly.interiors]:
            xy_ = list(ring.coords)
            for i in range(len(xy_) - 1):
                edge_use[_ekey(xy_[i], xy_[i + 1])] += 1

    n_oc = sum(c == "ocean" for _, c in faces)
    if n_oc == 0:
        raise RuntimeError("No ocean face found in this bbox.")
    print(f"{n_oc} ocean face(s), {len(faces) - n_oc} land face(s)")

    # ---------------- 3. Gmsh: mesh ALL faces (land + ocean) ----------------
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("domain2d")
    geo = gmsh.model.geo

    pts, lines, kind = {}, {}, {}

    def P(x, y):
        key = (round(x, 4), round(y, 4))
        if key not in pts:
            pts[key] = geo.addPoint(x, y, 0.0, lc_coarse)
        return pts[key]

    def L(a, b):
        if (a, b) in lines:
            return lines[(a, b)]
        if (b, a) in lines:
            return -lines[(b, a)]
        lines[(a, b)] = geo.addLine(a, b)
        return lines[(a, b)]

    def ring_curves(ring):
        xy_ = list(ring.coords)
        ids = [P(x, y) for x, y in xy_]
        out = []
        for i in range(len(ids) - 1):
            if ids[i] == ids[i + 1]:
                continue
            c = L(ids[i], ids[i + 1])
            kind[abs(c)] = ("boundary" if edge_use[_ekey(xy_[i], xy_[i + 1])] == 1
                            else "shoreline")
            out.append(c)
        return out

    surfs = {"ocean": [], "land": []}
    for poly, cls in faces:
        loops = [geo.addCurveLoop(ring_curves(r))
                 for r in [poly.exterior, *poly.interiors]]
        surfs[cls].append(geo.addPlaneSurface(loops))
    geo.synchronize()

    shore_curves = sorted(c for c, k in kind.items() if k == "shoreline")

    if shore_curves:
        f = gmsh.model.mesh.field
        f.add("Distance", 1)
        f.setNumbers(1, "CurvesList", shore_curves)
        f.setNumber(1, "Sampling", 100)
        f.add("Threshold", 2)
        f.setNumber(2, "InField", 1)
        f.setNumber(2, "SizeMin", lc_fine)
        f.setNumber(2, "SizeMax", lc_coarse)
        f.setNumber(2, "DistMin", dist_min)
        f.setNumber(2, "DistMax", dist_max)
        f.setAsBackgroundMesh(2)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    # ---------------- 4. 2D mesh -> numpy ----------------
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = coords.reshape(-1, 3)
    order = np.argsort(tags)
    tags, xyz = tags[order], xyz[order]

    def tris_of(surface_list):
        arr = [gmsh.model.mesh.getElementsByType(2, s)[1].reshape(-1, 3)
               for s in surface_list]
        arr = [a for a in arr if len(a)]
        return np.searchsorted(tags, np.concatenate(arr))

    tri_oc = tris_of(surfs["ocean"])
    tri_ld = tris_of(surfs["land"]) if surfs["land"] else np.empty((0, 3), int)
    tri = np.concatenate([tri_oc, tri_ld])
    is_ocean = np.concatenate([np.ones(len(tri_oc), bool),
                               np.zeros(len(tri_ld), bool)])
    M = len(tri)

    shore_edges = [gmsh.model.mesh.getElementsByType(1, c)[1].reshape(-1, 2)
                   for c in shore_curves]
    shore_edges = (np.searchsorted(tags, np.concatenate(shore_edges))
                   if shore_edges else np.empty((0, 2), int))
    gmsh.clear()

    N = len(tags)
    xy = xyz[:, :2]

    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    area2 = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    flip = area2 < 0
    tri[flip] = tri[flip][:, ::-1]                    # CCW in (x, y)

    # ---------------- 5. Topography at nodes ----------------
    lon_n, lat_n = inv_tr.transform(xy[:, 0], xy[:, 1])
    zt = interp(np.column_stack([lat_n, lon_n]))

    in_oc = np.zeros(N, bool); in_oc[tri[is_ocean].ravel()] = True
    in_ld = np.zeros(N, bool); in_ld[tri[~is_ocean].ravel()] = True
    zt = np.where(in_oc & in_ld, 0.0,
         np.where(in_oc, np.minimum(zt, 0.0), np.maximum(zt, 0.0)))

    A, B = 0, N                                       # A = topography, B = sea surface
    t_oc, t_ld = tri[is_ocean], tri[~is_ocean]
    seabed_i = t_oc + A
    land_i = t_ld + A
    sea_i = t_oc + B

    # ---------------- 6. Domain boundary ring (CCW) ----------------
    E = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    code = (np.minimum(E[:, 0], E[:, 1]).astype(np.int64) * N
            + np.maximum(E[:, 0], E[:, 1]))
    _, first, cnt = np.unique(code, return_index=True, return_counts=True)
    first = first[cnt == 1]
    fe = E[first]
    fe_ocean = is_ocean[first % M]

    def side_of(e):
        mlon = 0.5 * (lon_n[e[:, 0]] + lon_n[e[:, 1]])
        mlat = 0.5 * (lat_n[e[:, 0]] + lat_n[e[:, 1]])
        d = np.column_stack([np.abs(mlon - extent[0]), np.abs(mlon - extent[1]),
                             np.abs(mlat - extent[2]), np.abs(mlat - extent[3])])
        return np.argmin(d, axis=1)                   # 0 W, 1 E, 2 S, 3 N

    nxt = dict(zip(fe[:, 0].tolist(), fe[:, 1].tolist()))
    start = int(fe[0, 0])
    ring = [start]
    cur = nxt[start]
    while cur != start:
        ring.append(cur)
        cur = nxt[cur]
    ring = np.array(ring)
    if len(ring) != len(fe):
        raise RuntimeError("Domain boundary is not a single simple loop "
                           "(a shoreline probably touches the bbox at a single node).")

    ring_edges = np.column_stack([ring, np.roll(ring, -1)])
    sd = side_of(ring_edges)
    change = np.nonzero(sd != np.roll(sd, 1))[0]
    shift = int(change[0]) if len(change) else 0
    ring = np.roll(ring, -shift)                      # start at a corner
    ring_edges = np.column_stack([ring, np.roll(ring, -1)])
    ring_side = side_of(ring_edges)

    # ---------------- 7. Earth sides + bottom meshed by Gmsh ----------------
    wall_nodes, wall_tris, ring_b = mesh_earth_sides(
        xy[ring], zt[ring], ring_side, earth_depth, lc_deep, wall_growth)
    bot_nodes, bot_tri = mesh_bottom(ring_b, earth_depth, lc_deep)

    # water walls (ocean part of the boundary, seabed -> sea surface)
    wo = fe[fe_ocean]
    wa, wb = wo[:, 0], wo[:, 1]
    water_wall = np.concatenate([
        np.column_stack([wa + A, wb + A, wb + B]),
        np.column_stack([wa + A, wb + B, wa + B]),
    ])
    water_side = np.tile(side_of(wo), 2)

    shore_i = shore_edges + A

    # ---------------- 8. Assemble, merge coincident nodes ----------------
    nodes_all = np.vstack([
        np.column_stack([xy, zt]),                    # A
        np.column_stack([xy, np.zeros(N)]),           # B
        wall_nodes,
        bot_nodes,
    ])
    oW = 2 * N
    oB = 2 * N + len(wall_nodes)

    earth_wall_i = [(sd_, t + oW) for sd_, t in wall_tris]
    bottom_i = bot_tri + oB

    all_idx = np.concatenate(
        [seabed_i.ravel(), land_i.ravel(), sea_i.ravel(), bottom_i.ravel(),
         water_wall.ravel(), shore_i.ravel()]
        + [t.ravel() for _, t in earth_wall_i])
    used = np.unique(all_idx)

    key = np.round(nodes_all[used], 3) + 0.0          # +0.0 removes -0.0
    _, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    nodes_u = nodes_all[used][first_idx]
    mapping = np.full(len(nodes_all), -1, dtype=np.int64)
    mapping[used] = inv.ravel()

    def remap(t, side=None):
        t = mapping[t]
        ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
        return (t[ok], side[ok]) if side is not None else t[ok]

    seabed_t = remap(seabed_i)
    land_t = remap(land_i)
    sea_t = remap(sea_i)
    bottom_t = remap(bottom_i)
    water_wall_t, water_side = remap(water_wall, water_side)
    shore_l = mapping[shore_i]
    shore_l = shore_l[shore_l[:, 0] != shore_l[:, 1]]

    earth_walls = {}
    for sd_, t in earth_wall_i:
        earth_walls.setdefault(sd_, []).append(remap(t))
    earth_walls = {k: np.concatenate(v) for k, v in earth_walls.items()}

    # ---------------- 9. Watertightness check ----------------
    earth_parts = [seabed_t, land_t, bottom_t] + list(earth_walls.values())
    water_parts = [sea_t, seabed_t[:, ::-1], water_wall_t]
    print("open/non-manifold edges  earth:",
          count_bad_edges([p for p in earth_parts if len(p)]),
          "| water:", count_bad_edges([p for p in water_parts if len(p)]))

    # ---------------- 10. Write ----------------
    surfaces = [("seabed", seabed_t), ("land_surface", land_t),
                ("sea_surface", sea_t), ("earth_bottom", bottom_t)]
    for k, s in enumerate(SIDES):
        if k in earth_walls:
            surfaces.append((f"earth_wall_{s}", earth_walls[k]))
    for k, s in enumerate(SIDES):
        if np.any(water_side == k):
            surfaces.append((f"water_wall_{s}", water_wall_t[water_side == k]))
    surfaces = [(n, t) for n, t in surfaces if len(t)]

    gmsh.model.add("earth_and_sea")
    gm = gmsh.model
    for i, _ in enumerate(surfaces, start=1):
        gm.addDiscreteEntity(2, i)
    gm.mesh.addNodes(2, 1, np.arange(1, len(nodes_u) + 1), nodes_u.ravel())
    for i, (nm, t) in enumerate(surfaces, start=1):
        gm.mesh.addElementsByType(i, 2, [], (t + 1).ravel())
        g = gm.addPhysicalGroup(2, [i])
        gm.setPhysicalName(2, g, nm)

    if len(shore_l):
        j = len(surfaces) + 1
        gm.addDiscreteEntity(1, j)
        gm.mesh.addElementsByType(j, 1, [], (shore_l + 1).ravel())
        g = gm.addPhysicalGroup(1, [j])
        gm.setPhysicalName(1, g, "shoreline")

    print(f"{len(nodes_u)} nodes | "
          + ", ".join(f"{nm}: {len(t)}" for nm, t in surfaces))
    print(f"x [{nodes_u[:,0].min():.0f}, {nodes_u[:,0].max():.0f}] m, "
          f"y [{nodes_u[:,1].min():.0f}, {nodes_u[:,1].max():.0f}] m, "
          f"z [{nodes_u[:,2].min():.0f}, {nodes_u[:,2].max():.0f}] m")

    if gui:
        gmsh.fltk.run()
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(output_mesh)
    print(f"Mesh written to '{output_mesh}'.")
    gmsh.finalize()


if __name__ == "__main__":
    generate_earth_and_sea_shells(
        netcdf_path="/home/ulrich/work/Maule/GEBCO_23_Jun_2025_67aae380b8ec/gebco_2024_n-28.0_s-45.0_w-81.0_e-64.0.nc",
        bbox=[-79.0, -66.0, -41.0, -30.0],
        var_name="elevation",
        z_zero=-20.0,           # None to disable
        smooth_range=100.0,     # metres, None to disable
        smooth_sigma_m=1000.0,  # metres
        proj_str=PROJ_STR,
        coarsen_factor=2,
        lc_coarse=2000.0,
        lc_fine=500.0,
        earth_depth=400_000.0,
        lc_deep=50_000.0,
        output_mesh="earth_sea_shells.stl",
        gui=True,
    )
