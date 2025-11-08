import os
from typing import Collection
from itertools import product

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull
from sklearn.cluster import HDBSCAN


DECODER = np.load('decode.npy')

###########################################################
# Mostly needed for 'inference.ipynb' notebook
###########################################################
# This module is meant to facilitate inference and final evaluation:
#
#   - wrap model inference as a function
#   - compute volumetric maps and visualize them
#   - tools to compute DCC metric
#   - tool to compute DVO metric
#   - introduce train/evaluation loop
#   - analysis of results
###########################################################


def format_size(size: float | int):
        if size > 1024 * 1024 * 1024:
            size /= 1024 * 1024 * 1024
            return f'{round(size, 0)} Gb'
        elif size > 1024 * 1024:
            size /= 1024 * 1024
            return f'{round(size, 0)} Mb'
        elif size > 1024:
            size /= 1024
            return f'{round(size, 0)} Kb'
        else:
            return f'{round(size, 0)} b'


def inference(model: torch.nn.Module,
              scpdb_id: str,
              batch_size: int = 128,
              voxels_dir: str = 'Data/Voxels',
              inference_dir: str = 'Data/Inference',
              exist_ok: bool = True,
              device: str = 'cpu'):
    """
    Inference of a single scPDB entry
    The sub-grid scores are stored into <inference_dir> as <scpdb_id>.npy

    Args:
        model (torch.nn.Module): model to train
        scpdb_id (str): scPDB id to infere values for
        voxels_dir (str, optional): directory to take voxelized proteins from. Defaults to 'Data/Voxels'.
        inference_dir (str, optional): directory to store the volumetric map to. Defaults to 'Data/Inference'.
        exist_ok (bool, optional): allows to skip already infered entries. Defaults to True.
        device (str, optional): device to use Defaults to 'cpu'.
    """

    # Check the input
    assert isinstance(model, torch.nn.Module), f'model must be a subclass of torch.nn.Module, not {type(model)}'
    assert isinstance(scpdb_id, str), f'scpdb_id must be a str, not {type(scpdb_id)}'
    assert device in {'cpu', 'mps', 'cuda'}, f"unknown device '{device}', consider: 'cpu', 'mps', 'cuda'"
    assert isinstance(voxels_dir, str), f'voxels_dir must be str, not {type(voxels_dir)}'
    assert isinstance(inference_dir, str), f'inference_dir must be str, not {type(inference_dir)}'
    assert isinstance(batch_size, int), f'batch_size must be int, not {type(batch_size)}'
    assert 1 <= batch_size, f'batch_size must be positive, not {batch_size}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'

    voxel_size = 1

    # Create the path and check if already exists
    path = os.path.join(inference_dir, scpdb_id + '.npy')
    if os.path.isfile(path) and exist_ok:
        return np.load(path)

    # Load the factorized voxelization and get the true voxel map
    atoms_grid = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'atoms_grid.npy'))
    occupancy = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'occupancy.npy'))
    atoms = DECODER[atoms_grid].astype('float32')
    atoms[:,:,:,:-1] = atoms[:,:,:,:-1] * occupancy[..., None]
    atoms = torch.tensor(np.moveaxis(atoms, -1, 0), dtype=torch.float32)

    # Get subgrid size and sliding window stride
    sub_grid_size = 16 // voxel_size
    stride = max(1, 4 // voxel_size)
    nx = 1 + (atoms.shape[1] - sub_grid_size) // stride
    ny = 1 + (atoms.shape[2] - sub_grid_size) // stride
    nz = 1 + (atoms.shape[3] - sub_grid_size) // stride

    atoms = atoms[:, :(nx - 1) * stride + sub_grid_size, :(ny - 1) * stride + sub_grid_size, :(nz - 1) * stride + sub_grid_size].contiguous()

    stride_c, stride_x, stride_y, stride_z = atoms.stride()
    grids = atoms.as_strided(
        size=(nx, ny, nz, 8, sub_grid_size, sub_grid_size, sub_grid_size),
        stride=(stride_x * stride, stride_y * stride, stride_z * stride, stride_c, stride_x, stride_y, stride_z),
    )

    # reshape to (N, 8, 16, 16, 16)
    grids = grids.reshape(nx*ny*nz, 8, sub_grid_size, sub_grid_size, sub_grid_size)

    # Load model to device
    model.to(device)
    model.eval()

    scores = []

    for i in range(0, len(grids), batch_size):

        batch = grids[i:i+batch_size].to(device, non_blocking=True)

        with torch.no_grad():
            scores_ = model(batch)

        scores_ = torch.sigmoid(scores_.view(-1))
        scores.append(scores_.detach().cpu())

    scores = torch.cat(scores, dim=0)
    scores = scores.reshape(nx, ny, nz)

    # Save the result
    np.save(path, scores)
    return scores


def inference_complete(model: torch.nn.Module,
                       scpdb_ids: Collection[str],
                       batch_size: int = 128,
                       voxels_dir: str = 'Data/Voxels',
                       inference_dir: str = 'Data/Inference',
                       exist_ok: bool = True,
                       device: str = 'cpu'):
    """
    Inference of several scPDB entries
    The sub-grid scores are stored into <inference_dir> as <scpdb_id>.npy files

    Args:
        model (torch.nn.Module): model to train
        scpdb_id (str): scPDB id to infere values for
        voxels_dir (str, optional): directory to take voxelized proteins from. Defaults to 'Data/Voxels'.
        inference_dir (str, optional): directory to store the volumetric map to. Defaults to 'Data/Inference'.
        exist_ok (bool, optional): allows to skip already infered entries. Defaults to True.
        device (str, optional): device to use Defaults to 'cpu'.
    """

    # Check the input
    assert isinstance(model, torch.nn.Module), f'model must be a subclass of torch.nn.Module, not {type(model)}'
    assert isinstance(scpdb_ids, Collection), f'scpdb_ids must be a Collection, not {type(scpdb_ids)}'
    assert device in {'cpu', 'mps', 'cuda'}, f"unknown device '{device}', consider: 'cpu', 'mps', 'cuda'"
    assert isinstance(voxels_dir, str), f'voxels_dir must be str, not {type(voxels_dir)}'
    assert isinstance(inference_dir, str), f'inference_dir must be str, not {type(inference_dir)}'
    assert isinstance(batch_size, int), f'batch_size must be int, not {type(batch_size)}'
    assert 1 <= batch_size, f'batch_size must be positive, not {batch_size}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    voxel_size = 1

    # Report
    print('Inference')
    print(f'Number of entries: {len(scpdb_ids)}, voxel_size: {voxel_size}')
    print(f'Expected size of generated files: {format_size(len(scpdb_ids) * 25*1024 // (voxel_size ** 3))}')
    os.makedirs(inference_dir, exist_ok=True)

    skipped_existing = 0
    true_size = 0

    # Call inference for every entry
    for scpdb_id in tqdm(scpdb_ids, 'Inference...'):

        # Create the path and check if already exists
        path = os.path.join(inference_dir, scpdb_id + '.npy')
        if os.path.isfile(path) and exist_ok:
            skipped_existing += 1

        inference(model=model,
                  scpdb_id=scpdb_id,
                  batch_size=batch_size,
                  voxels_dir=voxels_dir,
                  inference_dir=inference_dir,
                  exist_ok=exist_ok,
                  device=device)

        true_size += os.path.getsize(path)

    # Report
    print(f'True size of generated files: {format_size(true_size)}')
    print(f'Entries skipped as existing: {skipped_existing}')
    return


def volumetric_map(scpdb_id: str,
                   radius: int = 2,
                   mode: str = 'mean',
                   inference_dir: str = 'Data/Inference') -> np.ndarray:
    """
    Expanding sub grid scores into a full-scale probabilities 3D heatmap

    Args:
        scpdb_id (str): scPDB id to infere values for
        radius (int, optional): radius (in number of voxels) to expend each score to. Defaults to 2.
        mode (str, optional): mode of aggregation of scores for a voxel (mean or max). Defaults to 'mean'.
        inference_dir (str, optional): directory to store the volumetric map to. Defaults to 'Data/Inference'.

    Returns:
        np.ndarray: volumetric map of probabilities per voxel
    """

    # Check the input
    assert isinstance(scpdb_id, str), f'scpdb_id must be a str, not {type(scpdb_id)}'
    assert isinstance(inference_dir, str), f'inference_dir must be str, not {type(inference_dir)}'
    assert isinstance(mode, str), f'mode must be str, not {type(mode)}'
    assert mode.lower() in ['mean', 'max'], f'mode must be from: mean / max, not "{mode}"'
    mode = mode.lower()
    assert isinstance(radius, int), f'radius must be int, not {type(radius)}'
    assert 1 <= radius, f'radius must be positive, not {radius}'
    voxel_size = 1

    # Extract the scores array
    path = os.path.join(inference_dir, scpdb_id + '.npy')
    scores = np.load(path)

    # Get the appropriate grid size andstride
    sub_grid_size = 16 // voxel_size
    stride = 4 // voxel_size

    # Initialize volumtric map and a counter array
    volume_map = np.zeros((
        (scores.shape[0] - 1) * stride + sub_grid_size,
        (scores.shape[1] - 1) * stride + sub_grid_size,
        (scores.shape[2] - 1) * stride + sub_grid_size
    ))
    counter = np.zeros_like(volume_map)

    # Iterate through each score
    for x, y, z in product(np.arange(scores.shape[0]), np.arange(scores.shape[1]), np.arange(scores.shape[2])):
        
        # Get the score
        score = scores[x,y,z]

        # Get the subgrid center on the volumetric map
        x = (sub_grid_size // 2) + x * stride
        y = (sub_grid_size // 2) + y * stride
        z = (sub_grid_size // 2) + z * stride

        # Update counter
        counter[x - radius:x + radius - 1, y - radius:y + radius - 1, z - radius:z + radius - 1] += 1

        # Get the view according to the radius
        volume_view = volume_map[x - radius:x + radius - 1, y - radius:y + radius - 1, z - radius:z + radius - 1]

        # Update the map in view span
        if mode == 'mean':
            volume_view += score
        else:
            volume_view[volume_view < score] = score

    # Average if required
    if mode == 'mean':
        counter[counter == 0] = 1
        volume_map = volume_map / counter

    return volume_map


def visualize_volumetric_map(scpdb_id: str,
                             show_binding_site: bool = True,
                             radius: int = 2,
                             mode: str = 'mean',
                             threshold: float = 0.5,
                             database_path: str = 'Data/database.csv',
                             inference_dir: str = 'Data/Inference',
                             voxels_dir: str = 'Data/Voxels',
                             scpdb_dir: str = 'Data/scPDB',
                             dpi: int = 300,
                             title: bool = True,
                             zoom: int = 0,
                             linewidth_default: float | int = 0.05,
                             linewidth_highlighted: float | int = 0.25,
                             save: str = None):
    """
    Allows to visualize a volumetric map obtained after the inference

    Args:
        scpdb_id (str): entry to visualize
        show_binding_site (bool, optional): also highlight the binding site. Defaults to True.
        radius (int, optional): radius (in number of voxels) to expend each score to. Defaults to 2.
        mode (str, optional): mode of aggregation of scores for a voxel (mean or max). Defaults to 'mean'.
        threshold (float, optional): minimum probability to consider a voxel as positive. Defaults to 0.5
        database_path (str, optional): path to the main database. Defaults to 'Data/database.csv'.
        inference_dir (str, optional): path to the folder with inferred scores. Defaults to 'Data/Atoms'.
        voxels_dir (str, optional): path to the folder with voxelized proteins. Defaults to 'Data/Voxels'.
        scpdb_dir (str, optional): path to the original scPDB folder. Defaults to 'Data/scPDB'.
        dpi (int, optional): resoluion. Defaults to 300.
        title (bool, optional): allows to turn off the title. Defaults to True.
        zoom (int, optional): allows to zoom in the plot. Defaults to 0.
        linewidth_default (float | int, optional): sets voxels edges width for default voxels. Defaults to 0.05.
        linewidth_highlighted (float | int, optional): sets voxels edges width for highlighted voxels. Defaults to 0.25.
        save (str, optional): path to save the plot to if any. Defaults to None.
    """
    
    # Check the input
    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'
    assert isinstance(scpdb_id, str), f'scpdb_id must be str, not {type(scpdb_id)}'
    assert isinstance(dpi, int), f'dpi must be int, not {type(dpi)}'
    assert isinstance(zoom, int), f'zoom must be int, not {type(zoom)}'
    assert isinstance(linewidth_default, float | int), f'linewidth_default must be float | int, not {type(linewidth_default)}'
    assert isinstance(linewidth_highlighted, float | int), f'linewidth_highlighted must be float | int, not {type(linewidth_highlighted)}'
    assert zoom >= 0, f'zoom must be non-negative, not {zoom}'
    assert linewidth_default >= 0, f'linewidth_default must be non-negative, not {linewidth_default}'
    assert linewidth_highlighted >= 0, f'linewidth_highlighted must be non-negative, not {linewidth_highlighted}'
    assert isinstance(threshold, float | int), f'threshold must be float (or 0 / 1), not {type(threshold)}'
    assert 0 <= threshold <= 1, f'threshold must be between 0 and 1, not {threshold}'
    assert isinstance(inference_dir, str), f'inference_dir must be str, not {type(inference_dir)}'
    assert isinstance(voxels_dir, str), f'voxels_dir must be str, not {type(voxels_dir)}'
    assert isinstance(scpdb_dir, str), f'scpdb_dir must be str, not {type(scpdb_dir)}'
    assert isinstance(show_binding_site, bool), f'show_binding_site must be bool, not {type(show_binding_site)}'
    assert isinstance(title, bool), f'title must be bool, not {type(title)}'
    assert isinstance(save, None | str), f'save must be None or str, not {type(save)}'
    assert isinstance(mode, str), f'mode must be str, not {type(mode)}'
    assert mode.lower() in ['mean', 'max'], f'mode must be from: mean / max, not "{mode}"'
    mode = mode.lower()
    voxel_size = 1

    # Getting the initial voxels to render as a semi-transparent base
    atoms_grid = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'atoms_grid.npy'))
    occupancy = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'occupancy.npy'))
    atoms = DECODER[atoms_grid].astype('float32')
    atoms[:,:,:,:-1] = atoms[:,:,:,:-1] * occupancy[..., None]
    
    # Getting the volumetric map of probabilities
    v_map = volumetric_map(scpdb_id=scpdb_id,
                           radius=radius,
                           mode=mode,
                           inference_dir=inference_dir)
    
    # Getting the true binding site
    centroid = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'site_center.npy'))
    in_site = np.zeros_like(v_map)
    in_site_voxels = np.full(v_map.shape, 'white', dtype='U5')


    if show_binding_site:
        site = open(os.path.join(scpdb_dir, scpdb_id, 'site.mol2'), 'r').read().split('@')[2]
        site_atoms = []
        for atom in site.split('\n')[1:-1]:
            x, y, z = list(filter(lambda x: x, atom.split(' ')))[2:5]
            site_atoms.append([float(x), float(y), float(z)])
        site_atoms = pd.DataFrame(site_atoms, columns=['X', 'Y', 'Z'])
        old_centroid = np.array(site_atoms[['X', 'Y', 'Z']].mean())
        shift = centroid - old_centroid
        for _, atom in site_atoms.iterrows():
            x = int(atom['X'] + shift[0]) // voxel_size
            y = int(atom['Y'] + shift[1]) // voxel_size
            z = int(atom['Z'] + shift[2]) // voxel_size
            in_site[x-1:x+2, y-1:y+2, z-1:z+2] = 1
            in_site_voxels[x-1:x+2, y-1:y+2, z-1:z+2] = 'black'

    # Plot
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    # Clip atoms grid and create indices
    atoms = atoms[:v_map.shape[0], :v_map.shape[1], :v_map.shape[2]]
    x, y, z = np.indices((v_map.shape[0] + 1, v_map.shape[1] + 1, v_map.shape[2] + 1)) * voxel_size

    x = x[6:-6, 6:-6, 6:-6]
    y = y[6:-6, 6:-6, 6:-6]
    z = z[6:-6, 6:-6, 6:-6]


    # Plot occupied voxels as pale green
    ax.voxels(x, y, z, atoms[6:-6, 6:-6, 6:-6, -1].astype(bool), alpha=0.1, facecolors='lightgreen', linewidth=linewidth_default, shade=True,
              edgecolors=in_site_voxels[6:-6, 6:-6, 6:-6])

    # Create grids of voxels of their colors
    colors = np.full(v_map.shape, 'white', dtype='U6')
    voxels = np.full(v_map.shape, False, dtype=bool)
    voxels[(v_map > threshold) & (atoms[..., -1] == 1)] = True
    colors[(v_map > threshold) & (atoms[..., -1] == 1)] = 'orange'

    # Plot thresholded voxels
    ax.voxels(x, y, z, voxels[6:-6, 6:-6, 6:-6], alpha=0.9, facecolors=colors[6:-6, 6:-6, 6:-6], linewidth=linewidth_highlighted, shade=True,
              edgecolors=in_site_voxels[6:-6, 6:-6, 6:-6])

    # Add the legend with channel colours
    legend = [
        Patch(facecolor='orange', edgecolor=None, label=f'Probability > {threshold}'),
        Patch(facecolor='white', edgecolor='k', label='True Binding Site')
    ]
    ax.legend(handles=legend,
            title=None,
            loc="upper left",
            frameon=True,
            fontsize=int(dpi/30),
            title_fontsize=int(dpi/25))

    # Set reasonable limits
    ax.set_xlim(6+zoom, v_map.shape[0]-6-zoom)
    ax.set_ylim(6+zoom, v_map.shape[1]-6-zoom)
    ax.set_zlim(6+zoom, v_map.shape[2]-6-zoom)

    # Turn off the grid and add a title
    ax.grid(False)
    ax.set_box_aspect([1,1,1])

    if not title:
        ax.set_axis_off()
    if title:
        ax.set_title(f"Volumetric Map of {scpdb_id}\n(probabilities for voxels to be in a binding site)", fontsize=int(dpi/20))
    if save is not None:
        fig.savefig(save, bbox_inches="tight", pad_inches=0)
    plt.show()


def _true_site_convex_hull_and_centroid(scpdb_id: str,
                                        scpdb_dir: str = 'Data/scPDB',
                                        voxels_dir: str = 'Data/Voxels') -> tuple[ConvexHull, np.ndarray]:
    """
    Helper function to get hull and centroid for the true pocket

    Args:
        scpdb_id (str): scPDB ID
        scpdb_dir (str, optional): folder with scPDB mol2 files. Defaults to 'Data/scPDB'.
        voxels_dir (str, optional): folder with voxelized proteins. Defaults to 'Data/Voxels'.

    Returns:
        tuple[ConvexHull, np.ndarray]: convex hull and centroid
    """

    # Read Mol2 file with site information (atoms)
    site = open(os.path.join(scpdb_dir, scpdb_id, 'site.mol2'), 'r').read().split('@')[2]
    site_atoms = []
    for atom in site.split('\n')[1:-1]:
        x, y, z = list(filter(lambda x: x, atom.split(' ')))[2:5]
        site_atoms.append([float(x), float(y), float(z)])
    site_atoms = np.array(site_atoms, dtype=float)

    # Get the 'old' center (from the raw data)
    old_centroid = site_atoms.mean(axis=0)

    # Get the 'new' center (after voxelization)
    centroid = np.load(os.path.join(voxels_dir, '1', scpdb_id, 'site_center.npy'))

    # Compute the 3D shift of coordinates (and apply it)
    shift = centroid - old_centroid
    site_atoms += shift

    # Get the convex hull (SciPy)
    convex_hull = ConvexHull(site_atoms)
    
    return centroid, convex_hull


def _predicted_convex_hull_and_centroid(scpdb_id: str,
                                        radius: int = 2,
                                        mode: str = 'mean',
                                        threshold: float = 0.5,
                                        inference_dir: str = 'Data/Inference') -> list[tuple[ConvexHull, np.ndarray]]:
    """
    Helper function to get hulls and centroida for predicted pockets

    Args:
        scpdb_id (str): scPDB ID
        radius (int, optional): radius of subgris center score influence. Defaults to 2.
        mode (str, optional): aggregation mode for scores (mean or max). Defaults to 'mean'.
        threshold (float, optional): minimum probability to consider a voxel positive. Defaults to 0.5.
        inference_dir (str, optional): folder with inferred scores. Defaults to 'Data/Inference'.

    Returns:
        list[tuple[ConvexHull, np.ndarray]]: convex hull and centroid for every found cluster
    """

    # Getting volumetric map
    v_map = volumetric_map(scpdb_id,
                           radius=radius,
                           mode=mode,
                           inference_dir=inference_dir)
    
    # Now I create an array of coordinates of voxels above the threshold
    # and cluster them using HDBSCAN
    points = np.argwhere(v_map > threshold)
    # points = points[(points[:,0] > 7) & (points[:,1] > 7) & (points[:,2] > 7) & (points[:,0] < v_map.shape[0] - 5) & (points[:,1] < v_map.shape[1] - 5) & (points[:,2] < v_map.shape[2] - 5)]

    pos_density = len(points)/(v_map.shape[0] * v_map.shape[1] * v_map.shape[2])
    # print(f'{100 * pos_density:.3g}')

    hdbscan = HDBSCAN(min_cluster_size=min(500, len(points) // 2), allow_single_cluster=True, n_jobs=-1)
    labels = hdbscan.fit_predict(points)

    # Extracting every cluster found as a separate predicted pocket
    hulls, centroids = [], []
    for cluster in range(0, np.max(labels) + 1):

        # Getting coordinates of 'positive' voxels
        pos_coords = points[np.argwhere(labels == cluster).flatten()] + (1 / 2)

        if len(pos_coords) < 4:
            continue

        # Get the center
        centroid = pos_coords.mean(axis=0)
        centroids.append(centroid)
        expend = (pos_coords > centroid) * 4
        expend[expend == 0] = -4
        pos_coords += expend

        # Get the convex hull (SciPy)
        convex_hull = ConvexHull(pos_coords)
        hulls.append(convex_hull)
    
    return centroids, hulls, pos_density


def _predicted_convex_hull_and_centroid_random(scpdb_id: str,
                                               threshold: float = 0.5,
                                               inference_dir: str = 'Data/Inference') -> list[tuple[ConvexHull, np.ndarray]]:
    """
    Helper function to get RANDOM hulls and centroids

    Args:
        scpdb_id (str): scPDB ID
        threshold (float, optional): minimum probability to consider a voxel positive. Defaults to 0.5.
        inference_dir (str, optional): folder with inferred scores. Defaults to 'Data/Inference'.

    Returns:
        list[tuple[ConvexHull, np.ndarray]]: convex hull and centroid for every found cluster
    """

    # Getting RANDOM volumetric map

    # Extract the scores array
    path = os.path.join(inference_dir, scpdb_id + '.npy')
    scores = np.load(path)

    # Get the appropriate grid size andstride
    sub_grid_size = 16
    stride = 4

    # Initialize RANDOM volumtric map and a counter array
    x, y, z  = (scores.shape[0] - 1) * stride + sub_grid_size, (scores.shape[1] - 1) * stride + sub_grid_size, (scores.shape[2] - 1) * stride + sub_grid_size
    v_map = np.zeros(x * y * z)
    v_map[:int(len(v_map) * (1 - threshold))] = 1
    np.random.shuffle(v_map)
    v_map = v_map.reshape((x, y, z))
    
    # Now I create an array of coordinates of voxels above the threshold
    # and cluster them using HDBSCAN
    points = np.argwhere(v_map > threshold)
    # pos_density = len(points)/(x * y * z)
    # print(f'{100 * pos_density:.3g}')

    hdbscan = HDBSCAN(min_cluster_size=min(500, len(points) // 2), allow_single_cluster=True, n_jobs=-1)
    labels = hdbscan.fit_predict(points)

    # Extracting every cluster found as a separate predicted pocket
    hulls, centroids = [], []
    for cluster in range(0, np.max(labels) + 1):

        # Getting coordinates of 'positive' voxels
        pos_coords = points[np.argwhere(labels == cluster).flatten()] + (1 / 2)

        # Get the center
        centroid = pos_coords.mean(axis=0)
        centroids.append(centroid)
        expend = (pos_coords > centroid) * 4
        expend[expend == 0] = -4
        pos_coords += expend

        # Get the convex hull (SciPy)
        convex_hull = ConvexHull(pos_coords)
        hulls.append(convex_hull)
    
    return centroids, hulls


def _hull_contains(hull: ConvexHull, points: np.ndarray) -> np.ndarray:
    """
    Helper function that takes a convex hull and an array of points and returns
    a boolean array (whether points are inside the hull or not)
    """

    A = hull.equations[:, :-1]
    b = hull.equations[:, -1:]

    return np.all(np.dot(points, A.T) + b.T <= 1e-12, axis=1)


def _hulls_overlap(hull_1: ConvexHull, hull_2: ConvexHull) -> tuple[int, int]:
    """
    Helper function that takes 2 convex hulls and returns:
    - number of voxels inside at least one of the hulls
    - number of voxels that are inside both hulls
    """

    # Get all the vertices
    points_1 = hull_1.points[hull_1.vertices]
    points_2 = hull_2.points[hull_2.vertices]
    all_points = np.vstack([points_1, points_2])

    # Find the boundaries and sample voxel centers
    mins = np.floor(all_points.min(axis=0))
    maxs = np.ceil(all_points.max(axis=0))
    dx = np.arange(mins[0], maxs[0]) + (1 / 2)
    dy = np.arange(mins[1], maxs[1]) + (1 / 2)
    dz = np.arange(mins[2], maxs[2]) + (1 / 2)

    # Check what voxels are inside each hull
    all_points = np.array(list(product(dx, dy, dz)))
    inside_hull_1 = _hull_contains(hull_1, all_points)
    inside_hull_2 = _hull_contains(hull_2, all_points)

    # Compute union and intersection
    union = np.sum((inside_hull_1 + inside_hull_2).astype(int))
    intersection =  np.sum((inside_hull_1 * inside_hull_2).astype(int))

    return union, intersection


def metrics(scpdb_id: str,
            radius: int = 2,
            mode: str = 'mean',
            threshold: float = 0.5,
            inference_dir: str = 'Data/Inference',
            scpdb_dir: str = 'Data/scPDB',
            voxels_dir: str = 'Data/Voxels') -> tuple[float, float]:
    """
    DCC and DVO metrics for a given scPDB ID

    Args:
        scpdb_id (str): scPDB ID
        radius (int, optional): radius of subgris center score influence. Defaults to 2.
        mode (str, optional): aggregation mode for scores (mean or max). Defaults to 'mean'.
        threshold (float, optional): minimum probability to consider a voxel positive. Defaults to 0.5.
        inference_dir (str, optional): folder with inferred scores. Defaults to 'Data/Inference'.
        scpdb_dir (str, optional): folder with scPDB mol2 files. Defaults to 'Data/scPDB'.
        voxels_dir (str, optional): folder with voxelized proteins. Defaults to 'Data/Voxels'.

    Returns:
        tuple[float, float]: DCC and DVO
    """

    # Get true and predicted hulls and centroids
    true_centroid, true_hull = _true_site_convex_hull_and_centroid(scpdb_id=scpdb_id,
                                                                   scpdb_dir=scpdb_dir,
                                                                   voxels_dir=voxels_dir)
    
    predicted_centers, predicted_hulls, pos_density = _predicted_convex_hull_and_centroid(scpdb_id=scpdb_id,
                                                                                          radius=radius,
                                                                                          mode=mode,
                                                                                          threshold=threshold,
                                                                                          inference_dir=inference_dir)

    
    # Choose the closest prediction by DCC
    closest_pocket = int(np.array([np.linalg.norm(predicted  - true_centroid) for predicted in predicted_centers]).argmin())
    predicted_centroid, predicted_hull = predicted_centers[closest_pocket], predicted_hulls[closest_pocket]

    # Compute DCC
    dcc = np.linalg.norm(true_centroid - predicted_centroid)

    # Compute DVO as Jaccard similarity
    union, intersection = _hulls_overlap(true_hull, predicted_hull)
    dvo = intersection / union if union != 0 else 0

    return dcc, dvo, pos_density


def random_metrics(scpdb_id: str,
                   threshold: float = 0.5,
                   inference_dir: str = 'Data/Inference',
                   scpdb_dir: str = 'Data/scPDB',
                   voxels_dir: str = 'Data/Voxels',
                   seed: int = 1234) -> tuple[float, float]:
    """
    DCC and DVO RANDOM metrics for a given scPDB ID

    Args:
        scpdb_id (str): scPDB ID
        threshold (float, optional): minimum probability to consider a voxel positive. Defaults to 0.5.
        inference_dir (str, optional): folder with inferred scores. Defaults to 'Data/Inference'.
        scpdb_dir (str, optional): folder with scPDB mol2 files. Defaults to 'Data/scPDB'.
        voxels_dir (str, optional): folder with voxelized proteins. Defaults to 'Data/Voxels'.
        seed (int, optional): random seed for reproducibility. Defaults to 1234.

    Returns:
        tuple[float, float]: DCC and DVO
    """

    # Get true and predicted hulls and centroids
    true_centroid, true_hull = _true_site_convex_hull_and_centroid(scpdb_id=scpdb_id,
                                                                   scpdb_dir=scpdb_dir,
                                                                   voxels_dir=voxels_dir)
    
    np.random.seed(seed)
    predicted_centers, predicted_hulls = _predicted_convex_hull_and_centroid_random(scpdb_id=scpdb_id,
                                                                                    threshold=threshold,
                                                                                    inference_dir=inference_dir)

    
    # Choose the closest prediction by DCC
    closest_pocket = int(np.array([np.linalg.norm(predicted  - true_centroid) for predicted in predicted_centers]).argmin())
    predicted_centroid, predicted_hull = predicted_centers[closest_pocket], predicted_hulls[closest_pocket]

    # Compute DCC
    dcc = np.linalg.norm(true_centroid - predicted_centroid)

    # Compute DVO as Jaccard similarity
    union, intersection = _hulls_overlap(true_hull, predicted_hull)
    dvo = intersection / union if union != 0 else 0

    return dcc, dvo


def analyse_metrics(metrics_path: str,
                    threshold: float = 0.5,
                    suptitle: str = None,
                    save: str = None):
    """
    Plot and aggregate final metrics

    Args:
        metrics_path (str): path to the final metrics CSV
        threshold (float, optional): threshold to choose. Defaults to 0.5.
        suptitle (str, optional): suptitle for the whole figure. Defaults to None.
        save (str, optional): path to save the plot to. Defaults to None
    """

    assert isinstance(metrics_path, str), f'metrics_path must be str, not {type(metrics_path)}'
    assert isinstance(threshold, float | int), f'threshold must be float (int), not {type(float)}'
    assert 0 <= threshold <= 1, f'threshold must be between 0 and 1, not {threshold}'
    assert isinstance(suptitle, str | None), f'suptitle must be str or None, not {type(suptitle)}'
    assert isinstance(save, str | None), f'save must be str or None, not {type(save)}'

    # Get the CSV and choose the threshold
    metrics = pd.read_csv(metrics_path, sep='\t')
    metrics = metrics.loc[metrics['Threshold'] == threshold]
    n_models = len(metrics['Model'].unique())
    report = []

    # Initialize a plot
    fig, axs = plt.subplots(nrows=2, ncols=1, dpi=300)
    dcc_ax = axs[0]
    dvo_ax = axs[1]

    # Iterate through models
    for i, model in enumerate(metrics['Model'].unique()):

        model_metrics = metrics.loc[metrics['Model'] == model]
        dcc_metrics = model_metrics['DCC']
        dvo_metrics = model_metrics['DVO']
        dvo_metrics = dvo_metrics[dvo_metrics != 0]
        min_dcc = 0

        # Construct a DCC Curve
        dcc_curve = []
        thresholds = []
        larger_50 = []

        for dcc_threshold in range(0, 250):

            dcc_threshold = min_dcc + dcc_threshold / 10
            percent = 100 * len(dcc_metrics[dcc_metrics <= dcc_threshold]) / len(dcc_metrics)
            dcc_curve.append(percent)
            thresholds.append(dcc_threshold)

            if percent >= 50:
                larger_50.append(dcc_threshold)

        report.append([model, threshold,
                       dcc_metrics.mean(), dcc_metrics.std(), f'{dcc_metrics.mean():.5g} ± {dcc_metrics.std() * 1.96 / len(dcc_metrics):.3g}', larger_50[0],
                       dvo_metrics.mean(), dvo_metrics.std(), f'{dvo_metrics.mean():.5g} ± {dvo_metrics.std() * 1.96 / len(dvo_metrics):.3g}'])

        # Plot DCC Curve
        dcc_ax.plot(thresholds, dcc_curve, label=model)
        dcc_ax.set_title('DCC Curve')
        dcc_ax.set_xlabel('DCC Threshold ($\AA$)')
        dcc_ax.set_ylabel('% of success')

        # Plot DVO boxplot
        dvo_ax.boxplot(100 * dvo_metrics[dvo_metrics > 0], positions=[i+1])
        dvo_ax.set_title('DVO Boxplot')
        dvo_ax.set_ylabel('DVO in %')

    dvo_ax.set_xticks(np.arange(1, n_models+1))
    dvo_ax.set_xticklabels(metrics['Model'].unique())
    dcc_ax.legend()
    dcc_ax.grid('--')

    # Show the plot (and save if required)
    plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle)
    if save is not None:
        fig.savefig(save)
    plt.show()

    # Report
    report = pd.DataFrame(report, columns=['Model', 'Threshold', 'DCC Mean', 'DCC SD', 'DCC CI (95%)', 'DCC 50%', 'DVO Mean', 'DVO SD', 'DVO CI (95%)'])
    report.set_index('Model', inplace=True, drop=True)
    report = report.transpose()

    return report


def analyse_scope(metrics_path: str = 'Data/final_metrics.csv',
                  main_csv_path: str = 'Data/database.csv',
                  model_name: str = 'cnn',
                  threshold: float = 0.9,
                  save: str = None):
    """
    Plot and aggregate final metrics

    Args:
        metrics_path (str, optional): path to the final metrics CSV. Defaults to Data/final_metrics.csv.
        main_csv_path (str, optional): path to the main CSV. Defaults to Data/database.csv.
        model_name (str, optional): name of the chosen model. Defaults to 'cnn'ArithmeticError.
        threshold (float, optional): threshold to choose. Defaults to 0.5.
        save (str, optional): path to save the plot to. Defaults to None
    """

    assert isinstance(metrics_path, str), f'metrics_path must be str, not {type(metrics_path)}'
    assert isinstance(main_csv_path, str), f'main_csv_path must be str, not {type(main_csv_path)}'
    assert isinstance(model_name, str), f'model_name must be str, not {type(model_name)}'
    assert isinstance(threshold, float | int), f'threshold must be float (int), not {type(float)}'
    assert 0 <= threshold <= 1, f'threshold must be between 0 and 1, not {threshold}'
    assert isinstance(save, str | None), f'save must be str or None, not {type(save)}'

    # Load the database and extract SCOPe identifiers
    database = pd.read_csv(main_csv_path, sep='\t')[['scPDB ID', 'SCOPe Chain Classes']]
    database.rename(columns={'SCOPe Chain Classes': 'scope'}, inplace=True)
    database['scope'] = database['scope'].apply(lambda x: list(set('.'.join(xx.split(' ')[1].split('.')[:2]) for xx in x.split(' / '))) if isinstance(x, str) else [])

    # Unravel the database: make each scPDB - SCOPe a separate line
    unraveled = []
    for i, row in database.iterrows():
        for family in row['scope']:
            unraveled.append([row['scPDB ID'], family])
    unraveled = pd.DataFrame(unraveled, columns=['ID', 'scope'])

    # Load the metrics
    metrics_df = pd.read_csv(metrics_path, sep='\t')
    metrics_df = metrics_df.loc[(metrics_df['Threshold'] == threshold) & (metrics_df['Model'] == model_name)]
    metrics_df.rename(columns={'scPDB ID': 'ID'}, inplace=True)
    metrics_df.drop(columns=['Threshold', 'Model'], inplace=True)

    # Merge metrics with an unraveled database (inner mode)
    unraveled = pd.merge(unraveled, metrics_df, how='inner')
    unraveled.drop(columns=['ID'], inplace=True)
    print(f'Different SCOPe families: {len(unraveled['scope'].unique())}')

    # Get mean, std and count for metrics grouped by SCOPe IDs
    families = unraveled.groupby(by='scope', as_index=False).mean()
    families_std = unraveled.groupby(by='scope', as_index=False).std()
    families_std.rename(columns={'DCC': 'std DCC', 'DVO': 'std DVO'}, inplace=True)
    counts = unraveled[['scope', 'DCC']].groupby(by='scope', as_index=False).count()
    counts.rename(columns={'DCC': 'N'}, inplace=True)
    families = pd.merge(families, counts, how='left')
    families = pd.merge(families, families_std, how='left')

    # Compute confidence intervals of mean for DCC and DVO
    families['DCC CI'] = 1.96 * families['std DCC'] / np.sqrt(families['N'])
    families['DCC upper'] = families['DCC'] + families['DCC CI']
    families['DCC lower'] = families['DCC'] - families['DCC CI']
    families['DVO CI'] = 1.96 * families['std DVO'] / np.sqrt(families['N'])
    families['DVO upper'] = families['DVO'] + families['DVO CI']
    families['DVO lower'] = families['DVO'] - families['DVO CI']
    families.loc[families['DCC lower'] < 0, 'DCC lower'] = 0
    families.loc[families['DVO lower'] < 0, 'DVO lower'] = 0
    families = families.loc[families['N'] > 4]
    print(f'SCOPe families with at least 5 examples: {len(families)}')
    families.set_index('scope', inplace=True)

    # Plot DCC bars
    ax_dcc = families[['DCC upper']].plot.bar(color=['lightgrey'], label=None)
    ax_dcc = families[['DCC']].plot(color=['black'], ax=ax_dcc, label=None, style='_')
    ax_dcc = families[['DCC lower']].plot.bar(color=['white'], ax=ax_dcc, label=None)
    ax_dcc.set_title('DCC Mean values (CI, p=.05) per SCOPe family\n(only families that have >= 5 examples in the test set)')
    ax_dcc.legend([])

     # Save if required
    if save is not None:
        ax_dcc.get_figure().savefig(save.split('.')[0] + '_dcc.' + save.split('.')[1])

    # Plot DVO bars
    ax_dvo = families[['DVO upper']].plot.bar(color=['lightgrey'], label=None)
    ax_dvo = families[['DVO']].plot(color=['black'], ax=ax_dvo, label=None, style='_')
    ax_dvo = families[['DVO lower']].plot.bar(color=['white'], ax=ax_dvo, label=None)
    ax_dvo.set_title('DVO Mean values (CI, p=.05) per SCOPe family\n(only families that have >= 5 examples in the test set)')
    ax_dvo.legend([])

    # Save if required
    if save is not None:
        ax_dvo.get_figure().savefig(save.split('.')[0] + '_dvo.' + save.split('.')[1])
