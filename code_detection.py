import rasterio
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge
from shapely.strtree import STRtree
from skimage.morphology import skeletonize, remove_small_objects, dilation
from rasterio.transform import from_origin
from PIL import Image
import os
import glob
from multiprocessing import Manager
import cv2
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import Manager, Value, Lock

# === PARAMÈTRES ===
input_folder = "C:\\Users\\DF-USER.DF-PORTABLE14\\Desktop\\a"   
output_folder = "C:\\Users\\DF-USER.DF-PORTABLE14\\Desktop\\b\\ilefrance"
foret_path = "foret.shp"
rpg_path = "PARCELLES_GRAPHIQUES.shp"
haies24_path = "haie_region\\haies_MIDI_PYRENEES.gpkg"
codes_interessants = ['18', '11', '19', '16', '28']

def traiter_dalle(jp2_path, compteur, total, lock):
    filename = os.path.splitext(os.path.basename(jp2_path))[0]
    output_shapefile = os.path.join(output_folder, f"{filename}.shp")

    if os.path.exists(output_shapefile):
        print(f"⏩ Déjà traité : {filename}.jp2 — on passe.")
        return

    print(f"🔄 Traitement de {filename}.jp2")
    t0 = time.time()

    # === Lecture image
    try:
        with rasterio.open(jp2_path) as src:
            image_np = src.read()
            transform = src.transform
            crs = src.crs
    except:
        image = Image.open(jp2_path).convert("L")
        image_np = np.array(image)
        transform = from_origin(0, 0, 1, 1)
        crs = None
    if crs is None:
        crs = "EPSG:2154"  # CRS par défaut à adapter selon ton cas

    print(f"📸 Chargement image : {time.time() - t0:.2f}s")
    t1 = time.time()

    # === Prétraitement
    if image_np.ndim == 3 and image_np.shape[0] > 1:
        image_np = np.mean(image_np, axis=0).astype(np.uint8)
        threshold_max = 40
    else:
        image_np = image_np[0] if image_np.ndim == 3 else image_np
        threshold_max = 74

    _, binary_mask = cv2.threshold(image_np.astype(np.uint8), threshold_max, 1, cv2.THRESH_BINARY_INV)
    binary = dilation(binary_mask, np.ones((3, 3), dtype=bool))
    binary = remove_small_objects(binary.astype(bool), min_size=50).astype(np.uint8)
    skeleton = skeletonize(binary)
    print(f"🧠 Prétraitement + squelette : {time.time() - t1:.2f}s")
    t2 = time.time()

    # === Extraction des lignes
    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if len(contour) > 2:
            coords = [tuple(transform * (pt[0][0], pt[0][1])) for pt in contour]
            line = LineString(coords)
            if line.is_valid and line.length > 5:
                lines.append(line)
    print(f"📏 Extraction lignes : {time.time() - t2:.2f}s")

    t3 = time.time()
    if not lines:
        print(f"⚠️ Aucun contour trouvé pour {filename}")
        return

    # On garde une référence explicite à la liste d'origine
    large_lines = [l for l in lines if l.length > 50]
    small_lines = [l for l in lines if l.length <= 50]

    filtered_lines = list(large_lines)

    if large_lines:
        # Création du STRtree AVEC référence explicite
        tree = STRtree(large_lines)

        for sl in small_lines:
            # Trouver les lignes proches
            idx_close = tree.query(sl.buffer(2), predicate='intersects')

            # Récupérer les vraies géométries (pas les index !)
            close_geoms = [large_lines[i] for i in idx_close]

            if not close_geoms or all(sl.length >= 0.3 * l.length for l in close_geoms):
                filtered_lines.append(sl)
    else:
        filtered_lines.extend(small_lines)

    print(f"📊 Lignes filtrées : {len(filtered_lines)}")

    if not filtered_lines:
        print(f"⚠️ Aucune ligne filtrée conservée pour {filename}")
        return

    merged = unary_union(filtered_lines)
    if isinstance(merged, MultiLineString):
        merged = linemerge(merged)

    if isinstance(merged, LineString):
        simplified_lines = [merged.simplify(5.0, preserve_topology=True)]
    elif isinstance(merged, MultiLineString):
        simplified_lines = [l.simplify(5.0, preserve_topology=True) for l in merged.geoms]
    else:
        simplified_lines = []

    if not simplified_lines:
        print(f"⚠️ Aucune ligne simplifiée pour {filename}")
        return

    haies_gdf = gpd.GeoDataFrame(geometry=simplified_lines, crs=crs)
    haies_gdf = haies_gdf[haies_gdf.geometry.length >= 9]

    if haies_gdf.empty:
        print(f"⚠️ Pas de haies valides pour {filename}")
        return
    print(f"🧹 Filtrage + simplification : {time.time() - t3:.2f}s")
    t4 = time.time()

    # === Suppression des haies dans la forêt
    if os.path.exists(foret_path):
        foret = gpd.read_file(foret_path)
        if foret.crs != haies_gdf.crs:
            foret = foret.to_crs(haies_gdf.crs)
        haies_gdf = gpd.overlay(haies_gdf, foret, how="difference")
    print(f"🌲 Exclusion forêt : {time.time() - t4:.2f}s")
    t5 = time.time()

    # === Filtrage par RPG
    if os.path.exists(rpg_path):
        rpg = gpd.read_file(rpg_path)
        if rpg.crs != haies_gdf.crs:
            rpg = rpg.to_crs(haies_gdf.crs)
        if "CODE_GROUP" in rpg.columns:
            rpg_sel = rpg[rpg["CODE_GROUP"].astype(str).isin(codes_interessants)]
            if not rpg_sel.empty:
                rpg_sel_buffered = rpg_sel.copy()
                rpg_sel_buffered["geometry"] = rpg_sel_buffered.buffer(5)
                haies_gdf = gpd.sjoin(haies_gdf, rpg_sel_buffered[["geometry"]], how="inner", predicate="within")
                haies_gdf = haies_gdf.drop(columns=["index_right"])
    print(f"🌾 Filtrage RPG : {time.time() - t5:.2f}s")
    t6 = time.time()

    # === Suppression haies existantes
    if os.path.exists(haies24_path):
        haies24 = gpd.read_file(haies24_path)
        if haies24.crs != haies_gdf.crs:
            haies24 = haies24.to_crs(haies_gdf.crs)

        if not haies24.empty and not haies_gdf.empty:
            # Réduire la zone d’intérêt à celle de l’image traitée
            minx, miny, maxx, maxy = haies_gdf.total_bounds
            haies24_clip = haies24.cx[minx:maxx, miny:maxy].copy()

            if not haies24_clip.empty:
                haies24_clip["geometry"] = haies24_clip.geometry.buffer(8)
                haies_gdf = gpd.overlay(haies_gdf, haies24_clip, how="difference")
                print(f"🧹 Haies existantes supprimées pour {filename}")
            else:
                print(f"⚠️ haies24 clip vide — aucun overlay appliqué")
        else:
            print(f"⚠️ haies24 ou haies_gdf vide — overlay ignoré")

    if haies_gdf.empty:
        print(f"⚠️ Plus aucune haie après filtres pour {filename}")
        return

    haies_gdf.to_file(output_shapefile, driver="ESRI Shapefile")
    print(f"✅ Exporté : {output_shapefile}")
    print(f"🕒 Temps total pour {filename} : {time.time() - t0:.2f}s\n")

    del haies24, haies24_clip
    import gc
    gc.collect()

    with lock:
        compteur.value += 1
        print(f"📦 {compteur.value}/{total} images traitées.")

def chunked(iterable, size):
    """Découpe une liste en lots de taille fixe."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

# === Traitement simple pour debug ===
if __name__ == "__main__":
    from itertools import islice

    def chunked(iterable, size):
        it = iter(iterable)
        while batch := list(islice(it, size)):
            yield batch

    os.makedirs(output_folder, exist_ok=True)
    jp2_files = glob.glob(os.path.join(input_folder, "*.jp2"))

    # 🔍 Filtrer les fichiers déjà traités
    non_traites = []
    for jp2 in jp2_files:
        filename = os.path.splitext(os.path.basename(jp2))[0]
        output_shapefile = os.path.join(output_folder, f"{filename}.shp")
        if not os.path.exists(output_shapefile):
            non_traites.append(jp2)

    total = len(non_traites)

    if total == 0:
        print("✅ Toutes les images ont déjà été traitées.")
        exit()

    print(f"🚀 Lancement du traitement de {total} images à traiter...")

    from multiprocessing import Manager

    with Manager() as manager:
        compteur = manager.Value('i', 0)
        lock = manager.Lock()

        for batch in chunked(non_traites,3):
            print(f"\n📦 Nouveau lot : {[os.path.basename(p) for p in batch]}")

            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(traiter_dalle, path, compteur, total, lock)
                    for path in batch
                ]
                for f in futures:
                    f.result()
