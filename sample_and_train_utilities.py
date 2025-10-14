import os
from typing import Collection
from collections import Counter
from itertools import product

import pandas as pd
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


DECODER = np.load('decode.npy')


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


def precompute_voxel_grids(list_of_entries: Collection[str],
                           database_path: str = 'Data/complete_database.csv',
                           atoms_dir: str = 'Data/Atoms',
                           voxels_dir: str = 'Data/Voxels',
                           voxel_size: int = 1,
                           exist_ok: bool = True):
    """
    Allows to generate voxel grids using occupancy score. 
    Entries from list_of_entries are processed, CSV with atom properties are taken from atoms_dir.

    After grid is ready, it is stores as a factorized pair of numpy arrays:
    - atoms_grid.npy: 3D array of int8 values. Each value encodes an 8-vector (can obtained through indexing a decode.npy array)
    - occupancy.npy: 3D array of occupancy scores per voxel
    + site_center.npy: contains a 3-vector of the binding site center

    All 3 files are stored in the fold:
    ./voxels_dir/voxel_size/entry

    Args:
        list_of_entries (Collection[str]): scPDB entries to voxelize
        database_path (str, optional): path to the main database. Defaults to 'Data/complete_database.csv'.
        atoms_dir (str, optional): path to a folder with atom properties CSVs. Defaults to 'Data/Atoms'.
        voxels_dir (str, optional): directory to store voxelization results. Defaults to 'Data/Voxels'.
        voxel_size (int, optional): the size of a voxel. Defaults to 1.
        exist_ok (bool, optional): allows to skip already voxelized entries. Defaults to True.
    """

    
    # Check the input
    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'
    assert isinstance(voxels_dir, str), f'voxels_dir must be str, not {type(voxels_dir)}'
    assert isinstance(atoms_dir, str), f'atoms_dir must be str, not {type(atoms_dir)}'
    assert isinstance(voxel_size, int), f'voxel_size must be int, not {type(voxel_size)}'
    assert 1 <= voxel_size, f'voxel_size must be positive, not {voxel_size}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    base_path = os.path.join(voxels_dir, str(voxel_size))
    os.makedirs(base_path, exist_ok=True)

    df = pd.read_csv(database_path, sep='\t')
    if list_of_entries is not None:
        assert isinstance(list_of_entries, Collection), f'list_of_entries must be a Collection, not {type(list_of_entries)}'
        for entry in list_of_entries:
            assert isinstance(entry, str), f'each entry in list_of_entries must be a str, not {type(entry)}'
        list_of_entries = list(set(list_of_entries))
    else:
        list_of_entries = df['scPDB IDs'].to_list()
    df = df.set_index("scPDB ID")

    # Van der Waals radii that I found on www.webelements.com
    vdw_radii = {'H': 1.1, 'Li': 1.82, 'C': 1.7, 'N': 1.55, 'O': 1.52, 
                'F': 1.47, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.1, 'P': 1.8,
                'S': 1.8, 'Cl': 1.75, 'K': 2.75, 'Ca': 2.31, 'Cu': 1.4,
                'Zn': 1.39, 'Se': 1.9, 'Br': 1.85, 'I': 1.98, 'Na': 2.27,
                'Mn': 2.45, 'Du': 0, 'Fe': 2.44}

    # Creating a decoder file
    # Each combination of binary channels (8) can be one-to-one mapped to an int8 value
    # So, instead of saving atom ids, which only make sense in the particular mol2 file, 
    # I will store one int8 value, which can be easily decoded to 8-vector of booleans
    atom_encodings = {tuple(int(v) for v in str(bin(i))[2:].rjust(8, '0')): i for i in range(256)}
    decipher_array = []
    for i in range(256):
        decipher_array.append(list(int(v) for v in str(bin(i))[2:].rjust(8, '0')))
    decipher_array = np.array(decipher_array)
    np.save('decode.npy', decipher_array)

    # Report
    print('Voxelization')
    print(f'Number of entries: {len(list_of_entries)}, voxel_size: {voxel_size}')
    print(f'Expected size of generated files: {format_size(len(list_of_entries) * 3*1024*1024 // (voxel_size ** 3))}')
    true_size = 0
    skipped_existing = 0

    # Go through every stored scPDB entry
    for name in tqdm(list_of_entries):

        # Skip if files are already generated
        if (exist_ok and
            os.path.isfile(os.path.join(base_path, name, 'atoms_grid.npy')) and 
            os.path.isfile(os.path.join(base_path, name, 'occupancy.npy')) and 
            os.path.isfile(os.path.join(base_path, name, 'site_center.npy'))):
            
            skipped_existing += 1
            continue

        row = df.loc[name]
        
        # Csv of protein atoms
        csv_path = os.path.join(atoms_dir, name + '.csv')
        csv = pd.read_csv(csv_path, sep='\t')

        # Raw centroid of binding site
        centroid = np.array([row['Site Center X'], row['Site Center Y'], row['Site Center Z']])
        
        # Making the coordinates start with 0,0,0 and zero-padding of size 8
        x_min = csv['X'].min() - 8
        y_min = csv['Y'].min() - 8
        z_min = csv['Z'].min() - 8
        csv['X'] -= x_min
        csv['Y'] -= y_min
        csv['Z'] -= z_min

        # Also I have to update the centroid in the new coordinate system
        centroid -= np.array([x_min, y_min, z_min])
        
        # Adding missing channels to atoms csv
        csv['Excluded Volume'] = 0

        # Creating a numpy array, where we can find atom properties vector by its index
        atom_vectors = csv[['Positive Ionizable', 'Negative Ionizable', 'Hydrophobic', 'Aromatic', 'Donor', 'Acceptor', 'Metal', 'Excluded Volume']]
        atom_vectors = atom_vectors.to_numpy()

        # The last row is all-zeros, because I will use -1 as default atom index
        atom_vectors = np.concatenate([atom_vectors, np.zeros((1,8))])

        # Adding 0-padding on another side
        x_max = int(np.ceil(csv['X'].max()) + 8)
        y_max = int(np.ceil(csv['Y'].max()) + 8)
        z_max = int(np.ceil(csv['Z'].max()) + 8)
        
        # Adding columns with elements and Van der Waals radii
        csv['Radius'] = [vdw_radii[element] for element in csv['Element']]

        # Initializing 3D grids:

        # Grid of occupancies: here we will store currently highest occupancy scores per voxel
        occupancies = np.zeros((x_max // voxel_size, y_max // voxel_size, z_max // voxel_size))

        # Grid of best atoms: here we will dynamically store atom indexes with highest scores per voxel
        best_atoms = np.zeros((x_max // voxel_size, y_max // voxel_size, z_max // voxel_size), dtype='int') - 1

        # Another grid of atoms, but here we will store encodings of atom properties instead of their indices
        best_atoms_enc = np.zeros((x_max // voxel_size, y_max // voxel_size, z_max // voxel_size), dtype='int')

        # Another grid of excluded volume (binary)
        excluded_volume = np.zeros((x_max // voxel_size, y_max // voxel_size, z_max // voxel_size), dtype='int')

        # Looking through every atom
        for i, atom in csv.iterrows():

            element = atom['Element']

            # Skip the placeholder
            if element == 'Du':
                continue
            
            # Gettng properties, encoding, coordinates and the largest window that might require updating
            encoding = atom_encodings[tuple(atom_vectors[i].astype('int'))]
            radius = atom['Radius']
            coordinates = atom[['X', 'Y', 'Z']].to_numpy()

            in_voxel = np.floor(coordinates / voxel_size).astype('int')
            excluded_volume[in_voxel[0], in_voxel[1], in_voxel[2]] = 1
            effective_distance = 3 * radius
            update_window = 1 + (int(effective_distance - voxel_size /2) // voxel_size)
            update_window = int(np.ceil(np.ceil(radius / 1.26) / voxel_size))

            # Update window span
            dx = (np.arange(update_window * 2 + 1) + in_voxel[0] - update_window) * voxel_size + 0.5 * voxel_size
            dx_n = ((dx - 0.5 * voxel_size) // voxel_size).astype('int')
            dy = (np.arange(update_window * 2 + 1) + in_voxel[1] - update_window) * voxel_size + 0.5 * voxel_size
            dy_n = ((dy - 0.5 * voxel_size) // voxel_size).astype('int')
            dz = (np.arange(update_window * 2 + 1) + in_voxel[2] - update_window) * voxel_size + 0.5 * voxel_size
            dz_n = ((dz - 0.5 * voxel_size) // voxel_size).astype('int')

            # Creating a sub-cube with coordinates from the window
            X, Y, Z = np.meshgrid(dx, dy, dz, indexing="ij")
            sub_volume = np.stack([X, Y, Z], axis=-1)

            # Calculating occupancy scores
            radii = np.sum((sub_volume - coordinates) ** 2, axis=3)
            radii = radii.astype('float')
            occupancy = 1 - np.exp(- (radius ** 2 / radii) ** 6)

            # Compare found occupancies with currently best scores
            sub_cube_occupancies = occupancies[dx_n[0]:dx_n[-1]+1, dy_n[0]:dy_n[-1]+1, dz_n[0]:dz_n[-1]+1]
            sub_cube_atoms = best_atoms[dx_n[0]:dx_n[-1]+1, dy_n[0]:dy_n[-1]+1, dz_n[0]:dz_n[-1]+1]
            sub_cube_atoms_enc = best_atoms_enc[dx_n[0]:dx_n[-1]+1, dy_n[0]:dy_n[-1]+1, dz_n[0]:dz_n[-1]+1]
            better_mask = (sub_cube_occupancies < occupancy)

            # Update the occupancies where higher values were found
            # Also store the atom where new best score was recorded
            sub_cube_occupancies = sub_cube_occupancies * (1 - better_mask.astype('int'))
            sub_cube_occupancies = sub_cube_occupancies + occupancy * better_mask.astype('int')
            sub_cube_atoms[better_mask] = i
            sub_cube_atoms_enc[better_mask] = encoding

            # Write down to the main grids
            occupancies[dx_n[0]:dx_n[-1]+1, dy_n[0]:dy_n[-1]+1, dz_n[0]:dz_n[-1]+1] = sub_cube_occupancies
            best_atoms[dx_n[0]:dx_n[-1]+1, dy_n[0]:dy_n[-1]+1, dz_n[0]:dz_n[-1]+1] = sub_cube_atoms
            best_atoms_enc[dx_n[0]:dx_n[-1]+1, dy_n[0]:dy_n[-1]+1, dz_n[0]:dz_n[-1]+1] = sub_cube_atoms_enc


        # Store the factorized data
        best_atoms_enc = best_atoms_enc + excluded_volume
        os.makedirs(os.path.join(base_path, name), exist_ok=True)
        np.save(os.path.join(base_path, name, 'atoms_grid.npy'), best_atoms_enc.astype('int8'))
        np.save(os.path.join(base_path, name, 'occupancy.npy'), occupancies.astype('float32'))
        np.save(os.path.join(base_path, name, 'site_center.npy'), centroid.astype('float32'))
        true_size += os.path.getsize(os.path.join(base_path, name, 'atoms_grid.npy'))
        true_size += os.path.getsize(os.path.join(base_path, name, 'occupancy.npy'))
        true_size += os.path.getsize(os.path.join(base_path, name, 'site_center.npy'))

    print(f'True size of generated files: {format_size(true_size)}')
    print(f'Entries skipped as existing: {skipped_existing}')


def visualize_voxels(scpdb_id: str,
                     channels: Collection[str] = 'all',
                     show_binding_site: bool = True,
                     voxel_size: int = 1,
                     database_path: str = 'Data/database.csv',
                     voxels_dir: str = 'Data/Voxels',
                     scpdb_dir: str = 'Data/scPDB',
                     dpi: int = 300,
                     title: bool = True,
                     zoom: int = 0,
                     save: str = None):
    """
    Allows to visualize a protein voxels

    Args:
        scpdb_id (str): entry to visualize
        channels (Collection[str], optional): channels to show. Defaults to {'Positive Ionizable', 'Negative Ionizable', 'Hydrophobic', 'Aromatic', 'Donor', 'Acceptor', 'Metal'}.
        voxel_size (int, optional): choose the voxel size. Defaults to.
        show_binding_site (bool, optional): also highlight the binding site. Defaults to True.
        database_path (str, optional): path to the main database. Defaults to 'Data/database.csv'.
        atoms_dir (str, optional): path to the folder with atom CSVs. Defaults to 'Data/Atoms'.
        scpdb_dir (str, optional): path to the original scPDB folder. Defaults to 'Data/scPDB'.
        dpi (int, optional): resoluion. Defaults to 300.
        title (bool, optional): allows to turn off the title. Defaults to True.
        zoom (int, optional): allows to zoom in the plot. Defaults to 0.
        save (str, optional): path to save the plot to if any. Defaults to None.
    """
    
    # Check the input
    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'
    assert isinstance(scpdb_id, str), f'scpdb_id must be str, not {type(scpdb_id)}'
    assert isinstance(dpi, int), f'dpi must be int, not {type(dpi)}'
    assert isinstance(zoom, int), f'zoom must be int, not {type(zoom)}'
    assert isinstance(voxel_size, int), f'voxel_size must be int, not {type(voxel_size)}'
    assert isinstance(voxels_dir, str), f'atoms_dir must be str, not {type(voxels_dir)}'
    assert isinstance(scpdb_dir, str), f'scpdb_dir must be str, not {type(scpdb_dir)}'
    assert isinstance(show_binding_site, bool), f'show_binding_site must be bool, not {type(show_binding_site)}'
    assert isinstance(title, bool), f'title must be bool, not {type(title)}'
    assert isinstance(save, None | str), f'save must be None or str, not {type(save)}'
    assert isinstance(voxel_size, int), f'voxel_size must be int, not {type(voxel_size)}'
    assert voxel_size in {1, 2}, f'voxel_size can be 1 or 2, not {voxel_size}'
    assert isinstance(channels, Collection), f'channels must be a Collection, not {type(channels)}'
    possible_channels = ['Positive Ionizable', 'Negative Ionizable', 'Hydrophobic', 'Aromatic', 'Donor', 'Acceptor', 'Metal']
    if channels == 'all':
        channels = possible_channels
        channels_ids = list(range(8))
    else:
        for channel in channels:
            assert channel in possible_channels, f'unknown channel "{channel}", consider: {possible_channels}'
        channels_ids = [possible_channels.index(channel) for channel in channels]

    channel_colours = {'Positive Ionizable': 'blue', 'Negative Ionizable': 'red', 'Hydrophobic': 'grey', 'Aromatic': 'darkgrey', 'Donor': 'lightblue', 'Acceptor': 'pink', 'Metal': 'orange'}
    
    atoms_grid = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'atoms_grid.npy'))
    occupancy = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'occupancy.npy'))

    atoms = DECODER[atoms_grid].astype('float32')
    atoms[:,:,:,:-1] = atoms[:,:,:,:-1] * occupancy[..., None]
    centroid = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'site_center.npy'))
    in_site = np.zeros_like(occupancy)

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

    # Plot
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    # The following block is needed to make atom sizes look correct
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width_inch = bbox.width
    xrange = atoms.shape[0]
    points_per_angstrom = width_inch * 72 / (2 * xrange)

    all_triples = list(product(list(range(8, atoms.shape[0]-8)), list(range(8, atoms.shape[1]-8)), list(range(8, atoms.shape[2]-8))))
    all_triples = sorted(all_triples, key=lambda x: (x[0] - atoms.shape[0])**2 + (x[1] - atoms.shape[1])**2 + (x[2] - atoms.shape[2])**2, reverse=True)

    # plot every atom according to its properties, if show_binding_site, then also add borders to 
    # atoms, that are parts of a binding site
    for x, y, z in all_triples:

        if atoms[x, y, z, -1] == 1:

            ax.scatter(x, y, z, marker='h', facecolor='lightgreen', alpha=0.1,
                        s = (voxel_size * np.sqrt(2) * (1 + zoom // 3) * points_per_angstrom) ** 2)

        for channel, channel_id in zip(channels, channels_ids):
            if atoms[x, y, z, channel_id] > 10 ** (-3):
                ax.scatter(x, y, z, marker='h', color=channel_colours[channel],
                            alpha=atoms[x, y, z, channel_id],
                            s = (voxel_size * np.sqrt(2) * (1 + zoom // 3) * points_per_angstrom) ** 2,
                            edgecolors = None if show_binding_site and in_site[x, y, z] == 0 else 'black',
                            linewidths = 0 if show_binding_site and in_site[x, y, z] == 0 else 0.3)
                ax.scatter(x, y, z, marker='1', color='black',
                            alpha=atoms[x, y, z, channel_id],
                            linewidth = 0 if show_binding_site and in_site[x, y, z] == 0 else 0.3,
                            s = (voxel_size * np.sqrt(2) * (1 + zoom // 3) * points_per_angstrom) ** 2)

    # Add the legend with channel colours
    legend = [
        Patch(facecolor=color, edgecolor='k', label=name)
        for name, color in channel_colours.items() if name in channels
    ]
    ax.legend(handles=legend,
            title="Chemical Channels\n(border indicates that voxel\nbelongs to the binding site)",
            loc="upper left",
            frameon=True,
            fontsize=int(dpi/30),
            title_fontsize=int(dpi/25))

    # Set reasinable limits
    ax.set_xlim(8+zoom, atoms.shape[0]-8-zoom)
    ax.set_ylim(8+zoom, atoms.shape[1]-8-zoom)
    ax.set_zlim(8+zoom, atoms.shape[2]-8-zoom)

    # Turn off the grid and add a title
    ax.grid(False)
    ax.set_box_aspect([1,1,1])

    if not title:
        ax.set_axis_off()
    if title:
        ax.set_title(f'Entry {scpdb_id}, channels shown: {", ".join(channels)}', fontsize=int(dpi/20))
    if save is not None:
        fig.savefig(save)
    plt.show()






def get_samples_by_id(scpdb_id: str,
                      voxel_size: int = 1,
                      distance_measure: str = 'euclidean',
                      distance_threshold: int = 4,
                      channels_to_use: Collection[str] = 'all',
                      voxels_dir: str = 'Data/Voxels',
                      neg_to_pos_ratio: float | int = 1,
                      seed: int = 1234) -> list[tuple]:
    """
    Takes scPDB entry and generates subgrids from the voxelized protein.

    The size of a voxel, distance type, distance threshold, channels and negatives-to-positives ratio can be chosen

    Args:
        scpdb_id (str): scPDB ID
        voxel_size (int, optional): size of a voxel in angstrom (A). Defaults to 1.
        distance_measure (str, optional): how to estimate distance to the true site center. 
            Possible options: 'euclidean', 'manhattan', 'uniform', Defaults to 'euclidean'.
        distance_threshold (int, optional): at what distance from site center label subgrids as positive. Defaults to 4.
        channels_to_use (Collection[str], optional): what chemicalchannels to use. Defaults to 'all'.
        voxels_dir (str, optional): directory to take voxelized proteins from. Defaults to 'Data/Voxels'.
        neg_to_pos_ratio (float | int, optional): number of negative samples per 1 positive sample. Defaults to 1.
        seed (int, optional): random seed for reproducibility. Defaults to 1234.

    Returns:
        list[tuple]: list of samples as (4D array, binary label)
    """

    # Check the input
    assert isinstance(scpdb_id, str), f'scpdb_id must be str, not {type(scpdb_id)}'
    assert isinstance(voxels_dir, str), f'voxels_dir must be str, not {type(voxels_dir)}'
    assert isinstance(distance_measure, str), f'distance_measure must be str, not {type(distance_measure)}'
    allowed_measures = {'euclidean', 'manhattan', 'uniform'}
    assert distance_measure in allowed_measures, f'unknown distance_measure "{distance_measure}", consider: {allowed_measures}'
    assert isinstance(voxel_size, int), f'voxel_size must be int, not {type(voxel_size)}'
    assert 1 <= voxel_size, f'voxel_size must be positive, not {voxel_size}'
    assert isinstance(distance_threshold, int), f'distance_threshold must be int, not {type(distance_threshold)}'
    assert 1 <= distance_threshold, f'distance_threshold must be positive, not {distance_threshold}'
    assert isinstance(neg_to_pos_ratio, int | float), f'neg_to_pos_ratio must be int or float, not {type(neg_to_pos_ratio)}'
    assert 0 < neg_to_pos_ratio, f'neg_to_pos_ratio must be positive, not {neg_to_pos_ratio}'
    assert isinstance(channels_to_use, Collection), f'channels_to_use must be a Collection, not {type(channel_to_use)}'
    possible_channels = ['Positive Ionizable', 'Negative Ionizable', 'Hydrophobic', 'Aromatic', 'Donor', 'Acceptor', 'Metal', 'Excluded Volume']
    if channels_to_use == 'all':
        channels = list(range(8))
    else:
        for channel in channels_to_use:
            assert channel in possible_channels, f'unknown channel "{channel}", consider: {possible_channels}'
        channels = [possible_channels.index(channel) for channel in channels_to_use]
    
    # Load the factorized voxelization
    atoms_grid = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'atoms_grid.npy'))
    occupancy = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'occupancy.npy'))
    centroid = np.load(os.path.join(voxels_dir, str(voxel_size), scpdb_id, 'site_center.npy'))


    # Get subgrid size and sliding window stride
    sub_grid_size = 16 // voxel_size
    stride = max(1, 4 // voxel_size)

    # Get dimensions and number of sub grids
    dx, dy, dz = atoms_grid.shape
    nx = 1 + ((dx - sub_grid_size) // stride)
    ny = 1 + ((dy - sub_grid_size) // stride)
    nz = 1 + ((dz - sub_grid_size) // stride)

    # Make a 4D array of sub grids
    X, Y, Z = np.meshgrid(np.arange(nx) * stride, np.arange(ny) * stride, np.arange(nz) * stride, indexing="ij")
    sub_grids = np.stack([X, Y, Z], axis=-1)

    # Find subgrids with true labels
    sub_grids_centers = sub_grids + sub_grid_size / 2
    if distance_measure == 'euclidean':
        distances = np.sum((sub_grids_centers - centroid.reshape(1, 1, 1, -1)) ** 2, axis=-1)
        distance_threshold = distance_threshold ** 2
    elif distance_measure == 'manhattan':
        distances = np.sum(np.abs((sub_grids_centers - centroid.reshape(1, 1, 1, -1))), axis=-1)
    else:
        distances = np.max(np.abs((sub_grids_centers - centroid.reshape(1, 1, 1, -1))), axis=-1)
    true_labels = sub_grids[distances <= distance_threshold].reshape(-1, 3)
    false_labels = sub_grids[distances > distance_threshold].reshape(-1, 3)

    # Choose false subgrids
    n_false_labels = int(np.ceil(neg_to_pos_ratio * len(true_labels)))

    np.random.seed(seed)
    false_labels = false_labels[np.random.choice(np.arange(len(false_labels)), n_false_labels)]

    # Generate samples
    result = []
    for label in true_labels:

        # Take a slice of atoms and decode them into 4D grid
        atoms_sub_grid = atoms_grid[label[0]:label[0] + sub_grid_size, label[1]:label[1] + sub_grid_size, label[2]:label[2] + sub_grid_size]
        atoms = DECODER[:,channels][atoms_sub_grid].astype('float32')

        # Take a slice of occupancy scores and multiply by atom properties
        sub_occupancy = occupancy[label[0]:label[0] + sub_grid_size, label[1]:label[1] + sub_grid_size, label[2]:label[2] + sub_grid_size]
        if 'Excluded Volume' in channels_to_use:
            atoms[:,:,:,:-1] = atoms[:,:,:,:-1] * sub_occupancy[..., None]
        else:
            atoms = atoms * sub_occupancy[..., None]

        # Move channels dimension forward (as PyTorch requires)
        result.append(
            (np.moveaxis(atoms.astype('float32'), -1, 0), 1)
        )

    for label in false_labels:
        
        # Take a slice of atoms and decode them into 4D grid
        atoms_sub_grid = atoms_grid[label[0]:label[0] + sub_grid_size, label[1]:label[1] + sub_grid_size, label[2]:label[2] + sub_grid_size]
        atoms = DECODER[:,channels][atoms_sub_grid]

        # Take a slice of occupancy scores and multiply by atom properties
        sub_occupancy = occupancy[label[0]:label[0] + sub_grid_size, label[1]:label[1] + sub_grid_size, label[2]:label[2] + sub_grid_size]
        if 'Excluded Volume' in channels_to_use:
            atoms[:,:,:,:-1] = atoms[:,:,:,:-1] * sub_occupancy[..., None]
        else:
            atoms = atoms * sub_occupancy[..., None]

        # Move channels dimension forward (as PyTorch requires)
        result.append(
            (np.moveaxis(atoms.astype('float32'), -1, 0), 0)
        )

    return result


def get_samples_by_uniprot(uniprot_id: str,
                           fold: pd.DataFrame,
                           neg_to_pos_ratio: float | int = 1,
                           augmented: bool = False,
                           voxel_size: int = 1,
                           distance_measure: str = 'euclidean',
                           distance_threshold: int = 4,
                           channels_to_use: Collection[str] = 'all',
                           voxels_dir: str = 'Data/Voxels',
                           seed: int = 1234) -> list[tuple]:
    """
    Takes Uniprot ID and a fold DataFrame and generates subgrid samples.

    The size of a voxel, distance type, distance threshold, channels and negatives-to-positives ratio can be chosen.

    (augmented=True will produce additional samples)

    Args:
        uniprot_id (str): UniProt ID
        fold (pd.DataFrame): pandas DataFrame with fold, which uniprot_id belongs to
        neg_to_pos_ratio (float | int, optional): number of negative samples per 1 positive sample. Defaults to 1.
        augmented (bool, optional): use additional samples. Defaults to False.
        voxel_size (int, optional): size of a voxel in angstrom (A). Defaults to 1.
        distance_measure (str, optional): how to estimate distance to the true site center. 
            Possible options: 'euclidean', 'manhattan', 'uniform', Defaults to 'euclidean'.
        distance_threshold (int, optional): at what distance from site center label subgrids as positive. Defaults to 4.
        channels_to_use (Collection[str], optional): what chemicalchannels to use. Defaults to 'all'.
        voxels_dir (str, optional): directory to take voxelized proteins from. Defaults to 'Data/Voxels'.
        seed (int, optional): random seed for reproducibility. Defaults to 1234.
        

    Returns:
        list[tuple]: list of samples as (4D array, binary label, weight)
    """

    # Input check (brief, as other parameters will be checked later anyway)
    assert isinstance(uniprot_id, str), f'uniprot_id must be str, not {type(uniprot_id)}'
    assert isinstance(augmented, bool), f'uaugmented must be bool, not {type(augmented)}'
    assert isinstance(fold, pd.DataFrame), f'fold must be pd.DataFrame, not {type(fold)}'

    # Fold DataFrame is expected to have UniProt IDs as row indices
    row = fold.loc[uniprot_id]

    # Get all the entries required
    if augmented:
        all_entries = eval(row['Names'])
        count = row['Count']
    else:
        all_entries = [row['Main']]
        count = 1

    # Call get_samples_by_id for every entry and add weight
    result = []
    for entry in all_entries:
        sub_result = get_samples_by_id(scpdb_id=entry,
                                        voxel_size=voxel_size,
                                        distance_measure=distance_measure,
                                        distance_threshold=distance_threshold,
                                        channels_to_use=channels_to_use,
                                        voxels_dir=voxels_dir,
                                        neg_to_pos_ratio=neg_to_pos_ratio,
                                        seed=seed)
        result.extend([(r[0], r[1], 1/count) for r in sub_result])

    return result


def get_samples_by_fold(fold_id: int | str,
                        folds_dir: str = 'Data/Folds',
                        partially: float = 1,
                        neg_to_pos_ratio: float | int = 1,
                        augmented: bool = False,
                        voxel_size: int = 1,
                        distance_measure: str = 'euclidean',
                        distance_threshold: int = 4,
                        channels_to_use: Collection[str] = 'all',
                        voxels_dir: str = 'Data/Voxels',
                        seed: int = 1234) -> list[tuple]:
    """
    Takes fold name/index and a folds directory and generates subgrid samples.

    The size of a voxel, distance type, distance threshold, channels and negatives-to-positives ratio can be chosen.

    (augmented=True will produce additional samples)

    Args:
        fold_id (int | str): index or name of the fold (as it is called in folds dir)
        folds_dir (pd.DataFrame, optional): directory to find a fold in. Defaults to 'Data/Folds'.
        partially (int | float, optional): get only a fraction of a fold. Defaults to 1.
        neg_to_pos_ratio (float | int, optional): number of negative samples per 1 positive sample. Defaults to 1.
        augmented (bool, optional): to use additional samples. Defaults to False.
        voxel_size (int, optional): size of a voxel in angstrom (A). Defaults to 1.
        distance_measure (str, optional): how to estimate distance to the true site center. 
            Possible options: 'euclidean', 'manhattan', 'uniform', Defaults to 'euclidean'.
        distance_threshold (int, optional): at what distance from site center label subgrids as positive. Defaults to 4.
        channels_to_use (Collection[str], optional): what chemicalchannels to use. Defaults to 'all'.
        voxels_dir (str, optional): directory to take voxelized proteins from. Defaults to 'Data/Voxels'.
        seed (int, optional): random seed for reproducibility. Defaults to 1234.
        
    Returns:
        list[tuple]: list of samples as (4D array, binary label, weight)
    """

    # Input Check (brief, other parameters will checked later anyway)
    assert isinstance(fold_id, int | str), f'fold_id must be int or str, not {type(fold_id)}'
    assert isinstance(folds_dir, str), f'folds_dir must be str, not {type(folds_dir)}'
    assert isinstance(partially, int | float), f'partially must be float or int, not {type(partially)}'
    assert 0 < partially <= 1, f'partially must be between 0 and 1, not {partially}'

    # Read the fold CSV
    fold = pd.read_csv(os.path.join(folds_dir, str(fold_id) + '.csv'))
    fold.index = fold['Uniprot ID']

    # Generate samples per each uniprot (or until partial fraction is reached)
    result = []
    already_got = 0
    needed = partially * len(fold)
    for uniprot_id in fold.index:

        if already_got >= needed:
            break

        sub_result = get_samples_by_uniprot(uniprot_id=uniprot_id,
                                            fold=fold,
                                            neg_to_pos_ratio=neg_to_pos_ratio,
                                            augmented=augmented,
                                            voxel_size=voxel_size,
                                            distance_measure=distance_measure,
                                            distance_threshold=distance_threshold,
                                            channels_to_use=channels_to_use,
                                            voxels_dir=voxels_dir,
                                            seed=seed)
        result.extend(sub_result)
        already_got += 1

    return result


def train_evaluate(model: torch.nn.Module, 
                   train_loader: DataLoader,
                   test_loader: DataLoader,
                   epochs: int = 5,
                   pos_weight: float | int = 1,
                   patience: int = 3,
                   device: str = 'cpu') -> tuple:
    """
    Training-evaluation loop

    Args:
        model (torch.nn.Module): model to train
        train_loader (DataLoader):loader for train data
        test_loader (DataLoader): loader for test data
        epochs (int, optional): epochs to train for. Defaults to 5.
        pos_weight (float | int, optional): weight of positive samples. Defaults to 1.
        patience (int, optional): how many epochs with non-decresing test loss are tolerated before early stop is triggered. Defaults to 3.
        device (str, optional): device to use Defaults to 'cpu'.

    Returns:
        tuple: (train_loss_per_epoch, test_loss_per_epoch, true_per_epoch, predictions_per_epoch)
    """

    # Check the input
    assert isinstance(model, torch.nn.Module), f'model must be a subclass of torch.nn.Module, not {type(model)}'
    assert isinstance(train_loader, DataLoader), f'train_loader must be a torch DataLoader, not {type(train_loader)}'
    assert isinstance(test_loader, DataLoader), f'test_loader must be a torch DataLoader, not {type(test_loader)}'
    assert isinstance(epochs, int), f'epochs must be int, not {type(epochs)}'
    assert epochs > 0, f'epochs must be positive, not {type(epochs)}'
    assert isinstance(patience, int), f'patience must be int, not {type(patience)}'
    assert patience > 0, f'patience must be positive, not {type(patience)}'
    assert isinstance(pos_weight, int | float), f'pos_weight must be float or int, not {type(pos_weight)}'
    assert device in {'cpu', 'mps', 'cuda'}, f"unknown device '{device}', consider: 'cpu', 'mps', 'cuda'"


    # Load model to device
    model.to(device)

    # We will use Adam optimized
    optimizer = torch.optim.Adam(model.parameters())

    # We will use BCE with Logits as a loss function
    pos_weight = torch.tensor([pos_weight])
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    # Initialize logs
    train_loss_per_epoch = []
    test_loss_per_epoch = []
    predictions_per_epoch = []
    true_per_epoch = []

    epochs_no_improvement = 0
    best_test_loss = np.inf

    # Iterate for every epoch
    for epoch in range(epochs):

        train_loss = 0
        test_loss = 0

        with tqdm(total=len(train_loader), desc=f'Training Epoch {epoch+1}: ') as pbar:

            for i, (x, y, w) in enumerate(train_loader):
                
                # Load data to device
                x.to(device)
                y.to(device)
                
                # Forward pass
                y_hat = model(x)

                # Loss computation (with individual weights)
                loss = loss_function(y_hat.type(torch.float32),
                                     y.reshape(-1, 1).type(torch.float32))
                loss = (loss * w).mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Collecting loss
                train_loss += loss.detach().item()
                mean_loss = train_loss / ((i+1) * train_loader.batch_size)

                # Updating the progress bar
                pbar.update(1)
                pbar.set_description(f'Training Epoch {epoch+1}: mean loss: {mean_loss:.3g}')
            
            # Collecting logs
            train_loss_per_epoch.append(train_loss / len(train_loader))


        # Evaluation
        with torch.no_grad():

            y_true = []
            y_pred = []

            with tqdm(total=len(test_loader), desc=f'Testing Epoch {epoch+1}: ') as pbar:

                for i, (x, y, w) in enumerate(test_loader):

                    # Load data to device
                    x.to(device)
                    y.to(device)
                    
                    # Forward pass
                    y_hat = model(x)

                    # Loss computation (with individual weights)
                    loss = loss_function(y_hat.type(torch.float32),
                                         y.reshape(-1, 1).type(torch.float32))
                    loss = (loss * w).mean()

                    # Collecting loss, true labels and predictions
                    test_loss += loss.detach().item()
                    y_true.extend(y.flatten().tolist())
                    y_pred.extend(y_hat.detach().flatten().tolist())

                     # Updating the progress bar
                    pbar.update(1)
                    pbar.set_description(f'Testing Epoch {epoch+1}: mean loss: {test_loss / ((i+1) * test_loader.batch_size):.3g}')
                
                # Collecting logs
                test_loss_per_epoch.append(test_loss / len(test_loader))

        # Storing labels and predictions (after sigmoid) to analyse later
        true_per_epoch.append(np.array(y_true))
        predictions_per_epoch.append(1/(1 + np.exp(-np.array(y_pred))))

        # For now we will just use a threshold of 0.5 to get some idea
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = 1/(1 + np.exp(-y_pred))
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        # Use sklearn to see the progress in evaluation with each epoch
        print(f'\nSklearn Metrics for Epoch {epoch+1}')
        print('Accuracy:', classification_report(y_true, y_pred, output_dict=True)['accuracy'])
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=['true 0', 'true 1'], columns=['pred 0', 'pred 1'])
        print(cm)
        print()

        if test_loss_per_epoch[-1] < best_test_loss:
            epochs_no_improvement = 0
            best_test_loss = test_loss_per_epoch[-1]
        else:
            epochs_no_improvement += 1
        
        if epochs_no_improvement >= patience:
            break

    # Return losses per epoch and also true and predicted logits for future analysis of the test part
    return train_loss_per_epoch, test_loss_per_epoch, true_per_epoch, predictions_per_epoch


class SiteDataSet(Dataset):

    def __init__(self,
                use_folds: list[int],
                folds_dir: str = 'Data/Folds',
                reduction_factor: int = 3,
                partially: float = 1,
                augmented=False,
                neg_to_pos_ratio=1,
                voxel_size: int = 1,
                distance_measure: str = 'euclidean',
                distance_threshold: int = 4,
                channels_to_use: Collection[str] = 'all',
                voxels_dir: str = 'Data/Voxels',
                seed: int = 1234):
        """
        Creates a dataset using the provided list of folds and additional parameters:

        Args:
            use_folds (list[int]): list of fold names (as they are called in folds_dir)
            folds_dir (pd.DataFrame, optional): directory to find a fold in. Defaults to 'Data/Folds'.
            reduction_factor (int, optional): reduce the number of generated sample. Defaults to 3.
            partially (int | float, optional): get only a fraction of every fold. Defaults to 1.
            neg_to_pos_ratio (float | int, optional): number of negative samples per 1 positive sample. Defaults to 1.
            augmented (bool, optional): to use additional samples. Defaults to False.
            voxel_size (int, optional): size of a voxel in angstrom (A). Defaults to 1.
            distance_measure (str, optional): how to estimate distance to the true site center. 
                Possible options: 'euclidean', 'manhattan', 'uniform', Defaults to 'euclidean'.
            distance_threshold (int, optional): at what distance from site center to label subgrids as positive. Defaults to 4.
            channels_to_use (Collection[str], optional): what chemical channels to use. Defaults to 'all'.
            voxels_dir (str, optional): directory to take voxelized proteins from. Defaults to 'Data/Voxels'.
            seed (int, optional): random seed for reproducibility. Defaults to 1234.
            
        Returns:
            list[tuple]: list of samples as (4D array, binary label, weight)
        """

        assert isinstance(reduction_factor, int), f'reduction_factor must be int, not {type(reduction_factor)}'
        assert reduction_factor >= 1, f'reduction_factor must be positive, not {reduction_factor}'

        self.augmented = augmented
        self.neg_to_pos_ratio=neg_to_pos_ratio
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.channels_to_use = channels_to_use
        self.n_channels = 8 if self.channels_to_use == 'all' else len(self.channels_to_use)
        self.partially = partially
        self.reduction_factor = reduction_factor
        self.seed = seed

        self.folds = sorted(use_folds)

        self.data = []
        for fold in self.folds:

            self.data.extend(
                get_samples_by_fold(fold_id=fold,
                                    folds_dir=folds_dir,
                                    partially=partially,
                                    neg_to_pos_ratio=neg_to_pos_ratio,
                                    augmented=augmented,
                                    voxel_size=voxel_size,
                                    distance_measure=distance_measure,
                                    distance_threshold=distance_threshold,
                                    channels_to_use=channels_to_use,
                                    voxels_dir=voxels_dir,
                                    seed=seed)
            )

        self.data = self.data[::self.reduction_factor]

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]


class FlexibleCNN3D(torch.nn.Module):

    def __init__(self, dropout: float = 0.25, channel_size: int | float = 2):
        """
        This Model will only be used for hyperparameter search during the pilot run

        Dropout will be used as it is after convolution layers and with a factor of 2 between dense layers.

        Channel size will be used with a factor of 16.

        Args:
            dropout (float, optional): dropout. Defaults to 0.25.
            channel_size (int | float, optional): relative channel size. Defaults to 2.
        """

        # Check the input
        assert isinstance(dropout, float | int), f'dropout must be float or int, not {type(dropout)}'
        assert 0 <= dropout, f'dropout must be at least 0, not {dropout}'
        assert isinstance(channel_size, float | int), f'channel_size must be int or float, not {type(channel_size)}'
        assert 0 < channel_size, f'channel_size must be positive, not {channel_size}'
        
        super().__init__()

        # Store parameters
        self.dropout = dropout
        self.channel_size = channel_size


        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(8, int(16 * channel_size), kernel_size=8, padding='same'),
            torch.nn.ELU(),
            torch.nn.Conv3d(int(16 * channel_size), int(16 * (channel_size + 1)), kernel_size=4, padding='same'),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Dropout3d(dropout),
            torch.nn.Conv3d(int(16 * (channel_size + 1)), int(16 * (channel_size + 2)), kernel_size=4, padding='same'),
            torch.nn.ELU(),
            torch.nn.Conv3d(int(16 * (channel_size + 2)), int(16 * (channel_size + 3)), kernel_size=4, padding='same'),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Dropout3d(dropout),
            torch.nn.Flatten(),
            torch.nn.Linear(int(64 * 16 * (channel_size + 3)), 128),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout * 2),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
    


def hyperparameter_search(train_folds: Collection[int|str],
                         test_folds: Collection[int|str],
                         batch_sizes: Collection[int],
                         dropouts: Collection[int|float],
                         channels: Collection[int],
                         exist_ok: bool = True,
                         max_epochs: int = 100,
                         patience: int = 5,
                         reduction_factor: int = 2,
                         voxel_size: int = 1,
                         folds_dir: str = 'Data/Folds',
                         voxels_dir: str = 'Data/Voxels',
                         pilot_dir: str = 'Data/CV/pilot',
                         seed: int = 1234):
    
    """
    Allows to run several pilot training loops to find the best hyperperameters for Batch Size, DropOut and number of channels.

    Args:
        train_folds (Collection[int|str]): collection of fold names (as they are called in folds_dir) to be used as a train set.
        test_folds (Collection[int|str]): collection of fold names (as they are called in folds_dir) to be used as a test set.
        batch_sizes (Collection[int]): collection of batch sizes to use as hyperparameters.
        dropouts (Collection[int|float]): collection of dropout values to use as hyperparameters.
        channels (Collection[int]): collection of channels to use as hyperparameters.
            The values of "channels" will be multiplied by 16 for the first convolution and will be increased with each layer.
        exist_ok (bool, optional): allows to skip runs which are already stored in the directory. Defaults to True.
        max_epochs (int, optional): maximum epochs to train for (can be less due to early stop). Defaults to 100.
        patience (int, optional): how many epochs with non-decresing test loss are tolerated before early start is triggered. Defaults to 5.
        reduction_factor (int, optional): reduce the number of generated sample. Defaults to 2.
        voxel_size (int, optional): size of a voxel in angstrom (A). Defaults to 1.
        folds_dir (pd.DataFrame, optional): directory to find a fold in. Defaults to 'Data/Folds'.
        voxels_dir (str, optional): directory to take voxelized proteins from. Defaults to 'Data/Voxels'.
        pilot_dir (str, optional): directory to store the runs logs to. Defaults to 'Data/CV/pilot'.
        seed (int, optional): random seed for reproducibility. Defaults to 1234.
    """

    # Check the input
    assert isinstance(train_folds, Collection), f'train_folds must be Collection, not {type(train_folds)}'
    all_folds = [f.replace('.csv', '') for f in os.listdir(folds_dir)]
    for fold in train_folds:
        assert str(fold) in all_folds, f'unknown fold "{fold}", consider: {all_folds}'

    assert isinstance(test_folds, Collection), f'test_folds must be Collection, not {type(test_folds)}'
    for fold in test_folds:
        assert str(fold) in all_folds, f'unknown fold "{fold}", consider: {all_folds}'
    
    assert isinstance(batch_sizes, Collection), f'batch_sizes must be Collection, not {type(batch_sizes)}'
    for batch in batch_sizes:
        assert isinstance(batch, int), f'batch size must be int, not {type(batch)}'

    assert isinstance(dropouts, Collection), f'dropouts must be Collection, not {type(dropouts)}'
    for do in dropouts:
        assert isinstance(do, float | int), f'dropout must be float or int, not {type(do)}'
        assert 0 <= do < 0.5, f'dropout must be between 0 and 0.5, not {do}'

    
    assert isinstance(channels, Collection), f'channels must be Collection, not {type(channels)}'
    for ch in channels:
        assert isinstance(ch, int), f'channel must be int, not {type(ch)}'
        assert 1 <= ch, f'channel must be at least 1, not {ch}'

    assert isinstance(max_epochs, int), f'max_epochs must be int, not {type(max_epochs)}'
    assert max_epochs > 0, f'max_epochs must be positive, not {max_epochs}'
    assert isinstance(patience, int), f'patience must be int, not {type(patience)}'
    assert patience > 0, f'patience must be positive, not {patience}'

    assert isinstance(reduction_factor, int | float), f'reduction_factor must be int or float, not {type(reduction_factor)}'
    assert reduction_factor >= 1, f'reduction_factor must be at least 1, not {reduction_factor}'

    assert isinstance(voxel_size, int), f'voxel_size must be int, not {type(voxel_size)}'
    assert voxel_size >= 1, f'voxel_size must be at least 1, not {voxel_size}'

    assert isinstance(folds_dir, str), f'folds_dir must be str, not {type(folds_dir)}'
    assert isinstance(voxels_dir, str), f'voxels_dir must be str, not {type(voxels_dir)}'
    assert isinstance(pilot_dir, str), f'pilot_dir must be str, not {type(pilot_dir)}'

    assert isinstance(seed, int), f'seed must be int, not {type(seed)}'
    assert seed > 0, f'seed must be positive, not "{seed}'


    # Create train and test datasets
    pilot_train_set = SiteDataSet(
        use_folds=train_folds,
        folds_dir=folds_dir,
        reduction_factor=reduction_factor,
        partially=1,
        augmented=False,
        neg_to_pos_ratio=1,
        voxel_size=voxel_size,
        distance_measure='euclidean',
        distance_threshold=4,
        channels_to_use='all',
        voxels_dir=voxels_dir,
        seed=seed)


    pilot_test_set = SiteDataSet(
        use_folds=test_folds,
        folds_dir=folds_dir,
        reduction_factor=reduction_factor,
        partially=1,
        augmented=False,
        neg_to_pos_ratio=1,
        voxel_size=voxel_size,
        distance_measure='euclidean',
        distance_threshold=4,
        channels_to_use='all',
        voxels_dir=voxels_dir,
        seed=seed)


    # Check the required combinations
    for batch_size in batch_sizes:

        # Create loaders
        pilot_train_loader = DataLoader(pilot_train_set, batch_size=batch_size, shuffle=True)
        pilot_test_loader = DataLoader(pilot_test_set, batch_size=batch_size)

        for dropout in dropouts:

            for channel_size in channels:

                path = os.path.join(pilot_dir, f'bs_{batch_size}_do_{dropout}_cs_{channel_size}'.replace('.', ''))

                if os.path.isdir(path) and exist_ok:
                    print(f'Run: Batch Size: {batch_size}; Dropout: {dropout}; Channel Size: {channel_size} is already present.')
                    continue

                print(f'Batch Size: {batch_size}; Dropout: {dropout}; Channel Size: {channel_size}')

                # Initialize the model
                torch.random.manual_seed(seed)
                flexible_model = FlexibleCNN3D(dropout=dropout, channel_size=channel_size)

                # Run the training loop
                train_loss_per_epoch, test_loss_per_epoch, true_per_epoch, predictions_per_epoch = train_evaluate(
                    model=flexible_model,
                    train_loader=pilot_train_loader,
                    test_loader=pilot_test_loader,
                    epochs=max_epochs,
                    pos_weight=1,
                    patience=patience,
                    device='cpu')
                
                # Store the results of a run
                os.makedirs(path, exist_ok=True)
                open(os.path.join(path, 'train_loss.txt'), 'w').write("\n".join(str(loss) for loss in train_loss_per_epoch))
                open(os.path.join(path, 'test_loss.txt'), 'w').write("\n".join(str(loss) for loss in test_loss_per_epoch))
                np.save(os.path.join(path, 'true_labels.npy'), np.array(true_per_epoch))
                np.save(os.path.join(path, 'predictions.npy'), np.array(predictions_per_epoch))


def analyse_hyperparameters(dir: str = 'Data/CV/pilot') -> pd.DataFrame:
    """
    Allows to briefly analyse hyperparameters stored in dir

    Args:
        dir (str, optional): path to the folder with hyperparameters. Defaults to 'Data/CV/pilot'.

    Returns:
        pd.DataFrame (df of hyperparameters)
    """

    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'

    # Get all combination folders
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
    all_folders = sorted([f for f in os.listdir(dir) if f[0] != '.'])

    # Keep track of limits
    min_loss = np.inf
    min_test_loss = np.inf
    best_run = None
    max_loss = -1
    max_epochs = -1
    stats = []

    # Iterate through combination folders and extract the logs + compute ROC AUC
    for i, folder in enumerate(all_folders):
        
        params = folder.split('_')
        if len(params) != 6:
            continue

        run_dict = {
            'batch_size': int(params[1]),
            'dropout': float('0.' + params[3][1:]),
            'channels': int(params[5]),
            'train_losses': [float(l) for l in open(os.path.join(dir, folder, 'train_loss.txt'), 'r').read().split('\n') if l],
            'test_losses': [float(l) for l in open(os.path.join(dir, folder, 'test_loss.txt'), 'r').read().split('\n') if l],
            'true_labels': [epoch for epoch in np.load(os.path.join(dir, folder, 'true_labels.npy'))],
            'predictions': [epoch for epoch in np.load(os.path.join(dir, folder, 'predictions.npy'))],
        }


        run_dict.update({'total_epochs': len(run_dict['test_losses']),
                         'best_test_loss': min(run_dict['test_losses']),
                         'best_train_loss': min(run_dict['train_losses']),
                         'best_epoch': np.argmin(run_dict['test_losses']) + 1,
                         'best_epoch_auc': roc_auc_score(run_dict['true_labels'][np.argmin(run_dict['test_losses'])].astype('int'),
                                                         run_dict['predictions'][np.argmin(run_dict['test_losses'])])
                         })

        if min(run_dict['test_losses'] + run_dict['train_losses']) < min_loss:
            min_loss = min(run_dict['test_losses'] + run_dict['train_losses'])
        if min(run_dict['test_losses']) < min_test_loss:
            min_test_loss = min(run_dict['test_losses'])
            best_run = i
        if max(run_dict['test_losses'] + run_dict['train_losses']) > max_loss:
            max_loss = max(run_dict['test_losses'] + run_dict['train_losses'])
        if len(run_dict['test_losses']) > max_epochs:
            max_epochs = len(run_dict['test_losses'])

        stats.append(run_dict)

    # Plot the runs
    fig, axs = plt.subplots(ncols=3, nrows=(int(len(all_folders) % 3 > 0) + (len(all_folders) // 3)),
                            sharex=True, sharey=True,
                            figsize=(12, 12))

    for i, run_dict in enumerate(stats):

        ax = axs[i//3][i%3]

        # Plot train losses and test losses
        ax.plot(run_dict['train_losses'], c='orange', label='Train Loss')
        ax.plot(run_dict['test_losses'], c='grey', label='Test Loss')

        # Plot dashed lines of best epoch / best test loss
        color = 'black' if i != best_run else 'red'
        ax.plot([np.argmin(run_dict['test_losses']),np.argmin(run_dict['test_losses'])], [0, 1], c=color, linestyle='--', linewidth=0.5)
        ax.plot([0, max_epochs], [min(run_dict['test_losses']), min(run_dict['test_losses'])], c=color, linestyle='--', linewidth=0.5)

        # Set limits
        ax.set_xlim([0, max_epochs])
        ax.set_ylim([0, max_loss])

        # Set title, legend and appropriate epoch ticks
        ax.legend()
        ax.set_title(f'Batch size {run_dict["batch_size"]} dropout {run_dict["dropout"]}, channels {16 * run_dict["channels"]}')
        ax.set_xticks(np.arange(0, max_epochs, 5))
        ax.set_xticklabels([str(tick+1) for tick in ax.get_xticks()])

    # Add suptitle and show the result
    fig.suptitle('Train and Test losses per HP combination.\n(Early stop is triggered with patience 5; Loss is in units per sample)')
    plt.tight_layout()
    plt.show()

    stats = pd.DataFrame(stats)
    return stats
        

