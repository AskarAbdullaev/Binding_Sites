import requests
import os
import time
import random

from typing import Collection
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import regex as re

from tqdm import tqdm
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

###########################################################
# This module is meant to facilitate preliminary steps:
#
#   - load the necessary databases
#   - collect information from web-pages
#   - create a dataframe with all the entries
#   - split the data
#   - briefly analyse the data
#   - perform the first preprocessing and visualization
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


def download_scPDB(list_of_entries: Collection[str] = None,
                   files_to_load: Collection[str] = {'protein.mol2', 'site.mol2'},
                   skip_existing_files: bool = True,
                   version: str = 'all',
                   dir: str = 'Data/scPDB'):
    """
    Allows to download scPDB respository (completely or partially).
    If no entries provided, it s expected to find a .txt file with each line corresponding to an entry
    in the current directory.

    IMPORTANT:
    scPDB entries are like PDB IDs, but also have upload index / version, e.g. "20gs" -> "20gs_2"
    If only PDB IDs are provided, the code will try to find its uploads in scPDB, which might take more time.
    If version='all': all found uploads will be downloaded. (If version='latest': only the latest upload)

    In the directory (dir) a separate folder for each entry will be cerated. Folder name will be its scPDB name (with version)
    In each folder files corresponding to this entry will be stored.

    If skip_existing_files is True, only missing files will be downloaded.

    ATTENTION:
    The total size of scPDB database is > 15Gb. The major part is from protein.mol2 files.

    Files that failed to be downloaded will be stores in a 'failed_downloads.csv' in the current directory

    Args:
        list_of_entries (Collection[str], optional): list of scPDB entries or PDB IDs to download. Defaults to None.
        files_to_load (Collection[str], optional): types of files to download,
            possible types: 'protein.mol2', 'site.mol2', 'cavity.mol2', 'IPF.txt', 'interaction.mol2', 'ligand.mol2'. Defaults to {'protein.mol2', 'site.mol2'}.
        skip_existing_files (bool, optional): skip files that are already in the directory. Defaults to True.
        version (str, optional): what uploads to download from scPDB (only important when PDB IDs provided).
            Possible options: 'all', 'latest'. Defaults to 'all'.
        dir (str, optional): directory where to store the database. Defaults to 'Data/scPDB'.

    """
    
    # Check that the list of entries is of correct dtype
    if list_of_entries is not None:
        assert isinstance(list_of_entries, Collection), f'list_of_entries must be a Collection, not {type(list_of_entries)}'
        for entry in list_of_entries:
            assert isinstance(entry, str), f'each entry in list_of_entries must be a str, not {type(entry)}'
    else:
        with open('all_scpdb_entries.txt', 'r') as file:
            list_of_entries = file.read().split('\n')
            list_of_entries = list(filter(lambda x: x, list_of_entries))
    list_of_entries = set(list_of_entries)

    # Check if files to load are of correct dtype and exist
    all_possible_files = {'protein.mol2', 'site.mol2', 'cavity.mol2', 'IPF.txt', 'interaction.mol2', 'ligand.mol2'}
    expected_kbytes = {'protein.mol2': 900_000, 'site.mol2': 60_000, 'cavity.mol2': 6_000, 'IPF.txt': 1_000, 'interaction.mol2': 6_000, 'ligand.mol2': 6_000}
    expected_size = 0
    if files_to_load is not None:
        assert isinstance(files_to_load, Collection), f'files_to_load must be a Collection, not {type(files_to_load)}'
        files_to_load = set(files_to_load)
        for file in files_to_load:
            assert isinstance(file, str), f'each file in files_to_load must be a str, not {type(entry)}'
            assert file in all_possible_files, f'unknown file: {file}, consider: {all_possible_files}'
            expected_size += expected_kbytes[file]


    # Check the dtypes of other keywords
    assert isinstance(skip_existing_files, bool), f'skip_existing_files must be bool, not {type(skip_existing_files)}'
    assert isinstance(version, str), f'version must be a str, not {type(version)}'
    assert version in {'latest', 'all'}, f'version can only be "latest" or "all", not {version}'
    
    
    # Estimate the total size of the downloads
    expected_size *= len(list_of_entries)
    
    # Report
    print('Downloading scPDB files')
    print(f'Entries provided: {len(list_of_entries)}')
    print(f'Files: {files_to_load}')
    print(f'Expected size of the downloads: {format_size(expected_size)}')
        

    # Downloading
    base_path = 'http://bioinfo-pharma.u-strasbg.fr/scPDB/EXPORTENTRY='
    file_suffix = {'protein.mol2': 'PROT=',
                    'site.mol2': 'SITE=',
                    'cavity.mol2': 'CAVITY=',
                    'IPF.txt': 'IFP=',
                    'interaction.mol2': 'INTS=',
                    'ligand.mol2': 'LIG='}
    true_size = 0
    failed_entries = []
    skipped_existing = 0

    
    with tqdm(total=len(list_of_entries) * len(files_to_load), desc='Downloading...') as pbar:

        # Iterate through provided entries
        for entry in list_of_entries:

            # If entry number is explicitly provided: just use it...
            if '_' in entry:
                valid_versions = [entry]

            # ...otherwise try different versions
            else:
                valid_versions = []
                for v in range(1, 21):

                    path = base_path + file_suffix[sorted(files_to_load)[0]] + entry + '_' + str(v)
                    response = requests.get(path)
                    if response.status_code != 200 or not ('TRIPOS' in response.text):
                        continue
                    valid_versions.append(entry + '_' + str(v))

            # If no valid versions found - skip the entry
            if not valid_versions:
                failed_entries.append([entry, file])
                pbar.update(len(files_to_load))
                continue

            # Only take the lastest added entry if required
            if version == 'latest':
                valid_versions = [valid_versions[-1]]

            # Iterate through found URLs
            for entry_v in valid_versions:

                # Create a new folder (optionally)
                os.makedirs(f'{dir}/{entry_v}', exist_ok=True)

                # Iterate through files to load
                for file in files_to_load:

                    # Skip the step if allowed
                    if skip_existing_files:
                        if os.path.isfile(f'{dir}/{entry_v}/{file}'):
                            pbar.update(1)
                            skipped_existing += 1
                            continue
                    
                    url = base_path + file_suffix[file] + entry_v
                    response = requests.get(url)
                    if response.status_code != 200:
                        failed_entries.append([entry_v, file])
                        pbar.update(1)
                        continue
                    else:
                        text = response.text
                        with open(f'{dir}/{entry_v}/{file}', 'w') as f:
                            f.write(text)
                        true_size += os.path.getsize(f'{dir}/{entry_v}/{file}')

                    pbar.update(1)

    failed_entries = pd.DataFrame(failed_entries, columns=['entry', 'file'])
    failed_entries.to_csv('failed_downloads.csv', sep=',')
    print(f'Downloading finished (actual size: {format_size(true_size)}; skipped as existing: {skipped_existing})')
    print(f'Failed entries: {len(failed_entries.entry.unique())} (files: {len(failed_entries)})')


def download_SCOPe(version: str = '2.08',
                   exist_ok: bool = True,
                   dir: str = 'Data/SCOPe'):
    """
    Allows to dowload SCOPe classification of PDB entries (and their chains),
    the version of SCOPe is selectable, directory to store is selectable

    Args:
        version (str, optional): version of SCOPe to download. Recommended 2.08 or 2.07. Defaults to '2.08'.
        exist_ok (bool, optional): allows to skip if CSV is already in the directory. Defaults to True.
        dir (str, optional): directory to store the file to. Defaults to 'Data/SCOPe'.
    """

    # Check the validity of the input
    allowed_versions = {'2.08', '2.07', '2.06', '2.05', '2.04', '2.03', '2.02', '2.01',
                        '1.75', '1.73', '1.71', '1.69', '1.67', '1.65', '1.63', '1.61',
                        '1.59', '1.57', '1.55'}
    assert isinstance(version, str), f'version must be a str, not {type(version)}'
    assert version in allowed_versions, f'uknown version "{version}", consider: {allowed_versions}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'

    # Report
    print('Downloading SCOPe')
    print(f'Version: {version}')
    path = os.path.join(dir, version.replace('.', '_') + '.csv')
    print(f'Path: {path}')

    # Ensure that dir exists and skip if file is already there
    os.makedirs(dir, exist_ok=True)
    if os.path.isfile(path):
        print(f'Version is already in the directory ({format_size(os.path.getsize(path))})')
        return

    # Downloading
    base_url = 'https://scop.berkeley.edu/downloads/parse/dir.cla.scope.@version@-stable.txt'
    url = base_url.replace('@version@', version)
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        print(f'Unfortunately URL request was not successful, code = {response.status_code}')
        return

    # Process the data and store it as a csv file (through pandas)
    csv = response.text.split('\n')
    csv = list(filter(lambda x: x and x[0] != '#', csv))
    csv = [line.split('\t')[:4] for line in csv]
    csv = pd.DataFrame(csv, columns=['scope_id', 'pdb_id', 'chain', 'class'])
    csv['chain'] = csv['chain'].str.split(':').apply(lambda x: x[0])
    csv.to_csv(path)

    # Report
    print(f'Version saved to the directory ({format_size(os.path.getsize(path))})')

    return


def download_scPDB_web_pages(exist_ok: bool = True,
                             dir: str = 'Data/Pages',
                             scpdb_dir: str = 'Data/scPDB',
                             exclude_entries: Collection = None):
    """
    Allows to download source codes of scPDB web-pages for the
    entries stored in scpdb_dir (by default: 'Data/scPDB')

    The source codes are then stored in dir (by default: 'Data/Pages')
    as entry.txt files.

    IMPORTANT: the procedure might take 6-12 h depending on the Internet connection.

    Args:
        exist_ok (bool, optional): allows to skip already saved pages. Defaults to True.
        dir (str, optional): directory to store pages to. Defaults to 'Data/Pages'.
        scpdb_dir (str, optional): directory to take entries from. Defaults to 'Data/scPDB'.
        exclude_entries (Collection, optional): exclude some entries. Defaults to None.
    """

    def get_source_code(entry_id):
        """
        Helper function to fetch the source code
        """

        url = f"http://bioinfo-pharma.u-strasbg.fr/scPDB/SITE={entry_id}"
        r = requests.get(url)
        if r.status_code != 200:
            return r.status_code
        return r.text

    # Check the input
    assert isinstance(exclude_entries, Collection), f'exclude_entries must be a str, not {type(exclude_entries)}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
    assert isinstance(scpdb_dir, str), f'scpdb_dir must be str, not {type(scpdb_dir)}'
    os.makedirs(dir, exist_ok=True)

    all_entries = sorted(set([file.replace('\n', '') for file in os.listdir(scpdb_dir)]) - set(exclude_entries))

    # Report
    print('Downloading scPDB web pages')
    print(f'Entries provided (from {scpdb_dir}): {len(all_entries)}')
    print(f'Expected size of the downloads: {format_size(40_000 * len(all_entries))}')

    true_size = 0
    skipped_existing = 0
    failed_entries = []

    # Iterate through entries
    for entry in tqdm(all_entries, desc='Loading scPDB web pages...', total=len(all_entries)):

        # Create the path and check if it already exixts
        path = os.path.join(dir, f'{entry}.txt')
        if exist_ok and os.path.isfile(path):
            skipped_existing += 1
            continue
        
        # Wait for a bit to avoid request errors
        lag = random.random() / 5
        time.sleep(lag)

        # Fetch the source code and save it
        page = get_source_code(entry)
        if not isinstance(page, int) and len(page) < 500_000:
            with open(path, 'w') as f:
                f.write(page)
            true_size += os.path.getsize(path)
        else:
            failed_entries.append([entry])

    failed_entries = pd.DataFrame(failed_entries, columns=['entry'])
    failed_entries.to_csv('failed_web_pages.csv', sep=',')
    print(f'Downloading finished (actual size: {format_size(true_size)}; skipped as existing: {skipped_existing})')
    print(f'Failed entries: {len(failed_entries)}')


def parse_table(table):
    """
    Helper function to parse html table (through bs4 utility)
    """
    t = []
    for row in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in row.find_all(["td","th"])]
        t.append(cells)

    return t


def parse_header(parser) -> dict:
    """
    Helper function

    Extracts:
    - PDB ID
    - PDB URL
    - resolution
    - method
    - deposition date

    using bs4 utility.

    Returns:
        dict
    """

    # Main information is stored in the From_Line divs.
    form_lines = parser.find_all("div", class_="Form_Line")

    # I expect 2 such divs. If not - the page is not valid
    if len(form_lines) < 2:
        return None

    # In the first Form_Line there should be PDB ID and Resolution
    first_form = form_lines[0].find_all("div", class_="InpT")
    pdb_a = first_form[0].find("a")
    pdb_id = pdb_a.get_text(strip=True) if pdb_a else None
    pdb_link = pdb_a["href"] if pdb_a else None
    resolution = first_form[1].get_text(strip=True)
    resolution = re.sub(r"[^0-9.\-]", "", resolution)

    # In the second Form_Line there should be method and deposition date
    second_form = form_lines[1].find_all("div", class_="InpT")
    method = second_form[0].get_text(strip=True).replace("Experimental Method :", "").strip()
    date = second_form[1].get_text(strip=True).replace("Deposition Date :", "").strip()

    return {
        "PDB ID": pdb_id,
        "PDB URL": pdb_link,
        "Resolution": resolution,
        "Method": method,
        "Deposition Date": date
    }


def parse_scpdb_source_code(entry: str,
                            source_code: str,
                            scope_df: pd.DataFrame) -> dict:
    """
    Allows to extract information from scPDB source code for an entry.

    Args:
        entry (str): scPDB entry
        source_code (str): source code as a str (already loaded)
        scope_df (pd.DataFrame): dataframe of SCOPe classes

    Returns:
        dict (dictionary with parsed information)
    """

    # Use bs4 to create an html parser
    parser = BeautifulSoup(source_code, "html.parser")
    entry = entry.removesuffix('.txt')
    base_path = 'http://bioinfo-pharma.u-strasbg.fr/scPDB'

    # Get main information
    info = {'scPDB ID': entry}
    info.update(parse_header(parser))

    # Protein Section
    protein_div = parser.find("div", {"class": "BOX1", "id": "ProtAnnot"})
    protein_body = protein_div.find("div", {"class": "BOX1-body"})

    # There should be 2 sections with tables (main and composition)
    uniprot_div = protein_body.find("div", {"id": "unip"})
    protein_tables = uniprot_div.find_all("table")

    composition_div = protein_body.find("div", {"id": "bsitecompos"})
    protein_composition_tables = composition_div.find_all("table")

    # Ligand Section
    ligand_div = parser.find("div", {"class": "BOX1", "id": "LigAnnot"})
    ligand_body = ligand_div.find("div", {"class": "BOX1-body"})

    # There should be 2 sections with tables (main and position)
    ligand_properties = ligand_body.find("div", {"id": "ligProps"})
    ligand_tables = ligand_properties.find_all("table")

    ligand_position = ligand_body.find("div", {"id": "ligPos"})
    ligand_position_tables = ligand_position.find_all("table")

    # The expected tables are:
    #   for protein:
    uniprot_table = parse_table(protein_tables[0])
    chains_table = parse_table(protein_tables[1])
    site_composition_table = parse_table(protein_composition_tables[0])
    cavity_table_1 = parse_table(protein_composition_tables[1])
    cavity_table_2 = parse_table(protein_composition_tables[2])
    #   for ligand:
    main_ligand_table = parse_table(ligand_tables[0])
    ligand_atoms_table = parse_table(ligand_tables[1])
    ligand_mass_center_table = parse_table(ligand_position_tables[0])

    # Extracting info from tables
    #   from uniprot table: Name, TaxID, Organism, Reign, AC, ID, EC
    for row in uniprot_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'name' in category.lower():
            info.update({'Name': value})
        elif 'taxid' in category.lower():
            info.update({'TaxID': value})
        elif 'organism' in category.lower():
            info.update({'Organism': value})
        elif 'reign' in category.lower():
            info.update({'Reign': value})
        elif 'ac' in category.lower():
            info.update({'AC': value})
        elif 'id' in category.lower():
            info.update({'Uniprot ID': value})
        elif 'ec' in category.lower():
            info.update({'EC': value})

    #   from chain table: Percentage of Residues within binding site per Chain
    chains_with_percentages = []
    for row in chains_table[1:]:
        if len(row) != 2:
            continue
        chain, percent = row
        percent = percent.replace(' ', '').replace('%', '')
        chains_with_percentages.append(f'{chain} = {percent}')
    chains_with_percentages = ' / '.join(chains_with_percentages)
    info.update({'Percentage of Residues within binding site per Chain': chains_with_percentages})

    #   from site composition table: Metals, Water, Residues, B-Factor
    for row in site_composition_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'metals' in category.lower():
            info.update({'Metals in Binding Site': value})
        elif 'cofactors' in category.lower():
            info.update({'Cofactors in Binding Site': value})
        elif 'water' in category.lower():
            info.update({'Water Molecules in Binding Site': value})
        elif 'non standard' in category.lower():
            info.update({'Non Standard Amino Acids in Binding Site': value})
        elif 'standard' in category.lower():
            info.update({'Standard Amino Acids in Binding Site': value})
        elif 'residues' in category.lower():
            info.update({'Number of Residues in Binding Site': value})
        elif 'factor' in category.lower():
            info.update({'B-Factor of Binding Site': value})

    #   from cavity tables: Ligandability, Volume, Hydrophobic Residues %, Polar Residues %
    info.update({'Cavity Ligandability': cavity_table_1[1][0]})
    info.update({'Cavity Volume (A^3)': cavity_table_1[1][1]})
    info.update({'Cavity % Hydrophobic': cavity_table_2[1][0]})
    info.update({'Cavity % Polar': cavity_table_2[1][1]})

    #   from ligand table: Surface Areas, DrugBank ID, Weight, Forula, HET
    for row in main_ligand_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'polar' in category.lower():
            info.update({'Ligand Polar Surface Area (A^2)': re.sub(r"[^0-9.\-]", "", value)})
        elif 'buried' in category.lower():
            info.update({'Ligand Buried Surface Area (A^2)': re.sub(r"[^0-9.\-]", "", value)})
        elif 'drugbank' in category.lower():
            info.update({'Ligand DrugBank ID': value})
        elif 'weight' in category.lower():
            info.update({'Ligand Molecular Weight': re.sub(r"[^0-9.\-]", "", value)})
        elif 'formula' in category.lower():
            info.update({'Ligand Formula': value})
        elif 'het' in category.lower():
            info.update({'Ligand Het Code': value})
    
    #   from liagnd atoms table: Donors, Acceptors, Rings, Charges atoms, Rotatable bonds
    for row in ligand_atoms_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'acceptor' in category.lower():
            info.update({'Ligand H-Bond Acceptors': value})
        elif 'donor' in category.lower():
            info.update({'Ligand H-Bond Donors': value})
        elif 'aromatic' in category.lower():
            info.update({'Ligand Aromatic Rings': value})
        elif 'rings' in category.lower():
            info.update({'Ligand Rings': value})
        elif 'anion' in category.lower():
            info.update({'Ligand Anionic Atoms': value})
        elif 'cation' in category.lower():
            info.update({'Ligand Cationic Atoms': value})
        elif 'five' in category.lower():
            info.update({'Ligand Rule of Five Violation': value})
        elif 'rotatable' in category.lower():
            info.update({'Ligand Rotatable Bonds': value})

    #   from liagnd mass center table: center of masses of a ligand
    info.update({'Ligand Mass Center (X)': ligand_mass_center_table[1][0]})
    info.update({'Ligand Mass Center (Y)': ligand_mass_center_table[1][1]})
    info.update({'Ligand Mass Center (Z)': ligand_mass_center_table[1][2]})

    # Now we also add links to dowloadable files
    info.update({'scPDB URL': os.path.join(base_path, f"SITE={entry}")})
    info.update({'Protein Mol2 Download URL': os.path.join(base_path, f'EXPORTENTRY=PROT={entry}')})
    info.update({'Ligand Mol2 Download URL': os.path.join(base_path, f'EXPORTENTRY=LIG={entry}')})
    info.update({'Site Mol2 Download URL': os.path.join(base_path, f'EXPORTENTRY=SITE={entry}')})
    info.update({'IFP Download URL': os.path.join(base_path, f'EXPORTENTRY=IFP={entry}')})
    info.update({'Cavity Download URL': os.path.join(base_path, f'EXPORTENTRY=CAVITY={entry}')})
    info.update({'Interaction Download URL': os.path.join(base_path, f'EXPORTENTRY=INTS={entry}')})
    info.update({'All files Download URL': os.path.join(base_path, f'EXPORTENTRY=ALL={entry}')})

    # Lastly, add information from SCOPe classification table:
    scope_sub_df = scope_df.loc[scope_df['pdb_id'] == info['PDB ID'], ["chain", "class"]]
    if len(scope_sub_df) == 0:
        info.update({'SCOPe Chain Classes': ''})
    else:
        scope_sub_df['combined'] = scope_sub_df[['chain', 'class']].apply(lambda row: row['chain'] + ' ' + row['class'], axis=1)
        chain_classes = " / ".join(scope_sub_df['combined'].to_list())
        # print(chain_classes)
        info.update({'SCOPe Chain Classes': chain_classes})
    
    return info


def parse_scPDB_pages(list_of_pages: Collection[str] = None,
                      pages_dir: str = 'Data/Pages',
                      output_path: str = 'Data/database.csv',
                      scope_path: str = 'Data/SCOPe/2_08.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses source codes of scPDB weg pages and creates a dataframe. (Plus: a dataframe of entries which produced errors during parsing)

    Web pages source codes are expected to be already collected and stored in a pages_dir (by default - 'Data/Pages')
    The resulting dataframe will be stored as output_path (by default - 'Data/database.csv')
    Frame of errors will be stored in 'parsing_failed.csv'

    The function also needs a path to the SCOPe classification dataframe (by default - 'Data/SCOPe/2_08.csv')

    Args:
        list_of_pages (Collection[str], optional): list of .txt files (paths) with source codes. If not provided - the
            whole page_dir will be used. Defaults to None.
        pages_dir (str, optional): directory with .txt files with source codes. Defaults to 'Data/Pages'.
        output_path (str, optional): path to store the final result. Defaults to 'Data/database.csv'.
        scope_path (str, optional): path to the SCOPe classification df. Defaults to 'Data/SCOPe/2_08.csv'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (final dataframe, dataframe of error logs)
    """

    # Check the input
    if list_of_pages is not None:
        from_dir = False
        assert isinstance(list_of_pages, Collection), f'list_of_pages must be a Collection, not {type(list_of_pages)}'
        for entry in list_of_pages:
            assert isinstance(entry, str), f'each entry in list_of_pages must be a str, not {type(entry)}'
    else:
        from_dir = True
        list_of_pages = [file.replace('\n', '') for file in os.listdir(pages_dir)]
        list_of_pages = list(filter(lambda x: x, list_of_pages))
    list_of_pages = set(list_of_pages)
    assert isinstance(pages_dir, str), f'page_dir must be str, not {type(pages_dir)}'
    assert isinstance(output_path, str), f'output_path must be str, not {type(output_path)}'
    assert isinstance(scope_path, str), f'scope_path must be str, not {type(scope_path)}'
    
    df = []
    errors = []
    scope_df = pd.read_csv(scope_path)

    # Report
    print('Parsing scPDB web pages')
    print(f'Entries provided (from {pages_dir if from_dir else "provided list"}): {len(list_of_pages)}')

    # Iterate through pages
    for page in tqdm(list_of_pages, desc='Parsing Source Pages...', total=len(list_of_pages)):

        try:
            # Get information from the source code
            path = Path(f'Data/Pages/{page}')
            source_code = path.open().read()
            result = parse_scpdb_source_code(page, source_code, scope_df)
            df.append(result)
        
        except BaseException as e:

            # Register errors
            errors.append([page, e])

    result_df = pd.DataFrame(df)
    errors_df = pd.DataFrame(errors, columns=['entry', 'error'])

    # Report
    print(f'Successfully parsed: {len(result_df)}')
    print(f'Errors occured: {len(errors_df)}')
    print(f'SCOPe not found for: {len(result_df.loc[result_df['SCOPe Chain Classes'].isna()])}')

    result_df.to_csv(output_path, sep='\t')
    errors_df.to_csv('failed_parsing.csv', sep='\t')

    return result_df, errors_df


def analyse_current_database(database_path: str):
    """
    Briefly analyse the database

    Args:
        database_path (str): path to the database of current data (.csv)

    Returns:
        pd.DataFrame
    """

    def counter_to_str(counter_items: list[tuple]):
        return ', '.join(f'{key} ({value})' for key, value in counter_items)

    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'

    # Read current database
    current_database = pd.read_csv(database_path, sep='\t')
    current_database = current_database.drop(columns=[col for col in current_database.columns if 'unnamed' in col.lower()])

    # Aggregate columns
    statistics = [['Total Entries', len(current_database)]]
    statistics.append(['Unique PDB IDs', len(current_database['PDB ID'].unique())])
    statistics.append(['Unique Uniprot IDs', len(current_database['Uniprot ID'].unique())])
    statistics.append(['Mean Resolution (Å)', current_database['Resolution'].mean()])
    statistics.append(['Number of Unique Species', len(current_database['Organism'].unique())])
    statistics.append(['Most Common Species', counter_to_str(Counter(current_database['Organism'].to_list()).most_common(3))])
    statistics.append(['Number of Unique Reigns', len(current_database['Reign'].unique())])
    statistics.append(['Most Common Reign', counter_to_str(Counter(current_database['Reign'].to_list()).most_common(3))])

    all_scope_classes, all_metals, all_cofactors = [], [], []
    sites_with_metals, sites_with_cofactors = 0, 0
    for i, row in current_database[['SCOPe Chain Classes', 'Metals in Binding Site', 'Cofactors in Binding Site']].iterrows():

        chain_scope_classes = row['SCOPe Chain Classes']
        metals = row['Metals in Binding Site']
        cofactors = row['Cofactors in Binding Site']

        if isinstance(metals, str) and len(metals) > 0:
            all_metals.extend(metals.split(' '))
            sites_with_metals += 1
        if isinstance(cofactors, str) and len(cofactors) > 0:
            all_cofactors.extend(cofactors.split(' '))
            sites_with_cofactors += 1
        if isinstance(chain_scope_classes, str) and len(chain_scope_classes) > 0:
            all_scope_classes.extend([".".join(chain[2:].split('.')[:3]) for chain in chain_scope_classes.split(' / ')])


    statistics.append(['SCOPe classes present', len(set(all_scope_classes))])
    statistics.append(['Most Common SCOPe classes', counter_to_str(Counter(all_scope_classes).most_common(3))])
    statistics.append(['Binding Sites with Metals', sites_with_metals])
    statistics.append(['Most Common Metals in Binding Sites', counter_to_str(Counter(all_metals).most_common(3))])
    statistics.append(['Binding Sites with Cofactors', sites_with_cofactors])
    statistics.append(['Most Common Cofactors in Binding Sites', counter_to_str(Counter(all_cofactors).most_common(3))])

    for column in ['Number of Residues in Binding Site', 'Standard Amino Acids in Binding Site', 'Non Standard Amino Acids in Binding Site', 
                    'Water Molecules in Binding Site', 'Cavity Ligandability', 'Cavity Volume (A^3)', 
                    'Cavity % Hydrophobic', 'Cavity % Polar', 'Ligand Molecular Weight', 'Ligand Buried Surface Area (A^2)', 'Ligand Polar Surface Area (A^2)',
                    'Ligand H-Bond Acceptors', 'Ligand H-Bond Donors', 'Ligand Rings', 'Ligand Aromatic Rings', 'Ligand Anionic Atoms', 'Ligand Cationic Atoms',
                    'Ligand Rule of Five Violation', 'Ligand Rotatable Bonds']:
        statistics.append([column.replace('A^2', 'Å²').replace('A^3', 'Å³') + ' (mean)', current_database[column].mean()])
    
    
    # Create and display DataFrame
    statistics = pd.DataFrame(statistics, columns=['Metric', 'Value(s)'])
    statistics.set_index('Metric', inplace=True)

    pd.set_option('display.max_colwidth', 100)

    return statistics


def analyse_original_split(original_split_txt: str,
                            database_path: str):
    """
    Briefly analyse the original split of the data

    Args:
        original_split_txt (str): path to the original split (.txt)
        database_path (str): path to the database of current data (.csv)

    Returns:
        pd.DataFrame
    """

    assert isinstance(original_split_txt, str), f'original_split_txt must be str, not {type(original_split_txt)}'
    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'

    statistics = []

    # Read the original split from the SUpplementary materials of DeepSite paper
    deep_site_original = open(original_split_txt).read()
    deep_site_original = deep_site_original.split('\n')
    deep_site_original = deep_site_original[1::2]
    deep_site_original = {i: list_of_ids.split(', ') for i, list_of_ids in zip(range(10), deep_site_original)}

    # Read current database
    current_database = pd.read_csv(database_path, sep='\t')[['scPDB ID', 'PDB ID', 'Uniprot ID']]

    # Compute statistics
    all_uniprot_ids = set()
    for i, fold in deep_site_original.items():
        original_fold_length = len(fold)
        currently_in_db = len(set(fold) & set(current_database['PDB ID'].unique()))
        current_unique_uniprot = current_database.loc[current_database['PDB ID'].isin(fold), 'Uniprot ID']
        all_uniprot_ids |= set(current_unique_uniprot.unique())
        current_unique_uniprot = len(set(current_unique_uniprot.unique()))
        statistics.append([original_fold_length, currently_in_db, current_unique_uniprot])
    
    # Create a DataFrame
    statistics = pd.DataFrame(statistics, columns=['Original Length', 'Currently in scPDB', 'Currently unique Uniprot IDs'])
    statistics.index = pd.Index(np.arange(1,11), name='Fold')
    statistics.loc['Total'] = [statistics['Original Length'].sum(),
                                statistics['Currently in scPDB'].sum(),
                                len(all_uniprot_ids)]

    return statistics


def split_dataset(database_path: str,
                  folds_dir: str = 'Data/Folds',
                  test_fraction: float = 0.25,
                  n_folds: int = 10,
                  seed: int = 1234) -> pd.DataFrame:
    """
    Allows to split the data (provided through database_path) into a test set and
    N folds for cross-validation.

    Folds are stored to the folds_dir (Data/Folds by default) as csv files named 0.csv, 1.csv, ..., n.csv and test.csv
    Each file has columns:
    - Uniprot ID: unique ID not present in any other fold or test set
    - Main: scPDB ID which is the representative of the unique UniProt
    - scPDB IDs: list of all scPDB IDs with this UniProt ID (can be used for data augmentation later)
    - Count: number of scPDB IDs with this UniProt ID

    Args:
        database_path (str): path to the database df
        folds_dir (str, optional): directory to store the folds to. Defaults to 'Data/Folds'.
        test_fraction (float, optional): fraction of unique IDs for the test set. Defaults to 0.25.
        n_folds (int, optional): number of folds for cross-validation. Defaults to 10.
        seed (int, optional): random seed for reproduceability. Defaults to 1234.

    Returns:
        pd.DataFrame (updated database)
    """

    # Check the input
    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'
    assert isinstance(folds_dir, str), f'folds_dir must be str, not {type(folds_dir)}'
    assert isinstance(test_fraction, float), f'test_fraction must be float, not {type(test_fraction)}'
    assert 0 < test_fraction < 1, f'test_fraction must be between 0 and 1, not {test_fraction}'
    assert isinstance(n_folds, int), f'n_folds must be int, not {type(n_folds)}'
    assert 0 < n_folds, f'n_folds must be positive, not {n_folds}'
    assert isinstance(seed, int), f'seed must be int, not {type(seed)}'
    assert 0 < seed, f'seed must be positive, not {seed}'

    os.makedirs(folds_dir, exist_ok=True)

    np.random.seed(seed)

    # Read current database
    current_database = pd.read_csv(database_path, sep='\t')
    current_database = current_database.drop(columns=[col for col in current_database.columns if 'unnamed' in col.lower()])
    
    # Find the test fraction
    all_uniprot_ids = set(id_ for id_ in current_database['Uniprot ID'].unique() if isinstance(id_, str) and len(id_) > 0)
    ids_for_final_testing = int(len(all_uniprot_ids) * test_fraction)
    ids_for_final_testing = np.random.choice(np.arange(len(all_uniprot_ids)), ids_for_final_testing, replace=False)
    ids_for_final_testing = [sorted(all_uniprot_ids)[i] for i in ids_for_final_testing]

    # Find the IDs, which remain for CV
    ids_for_cv = all_uniprot_ids - set(ids_for_final_testing)

    # Find IDs per fold and the remainder
    ids_per_fold = len(ids_for_cv) // n_folds
    remainder = len(ids_for_cv) % n_folds

    # Report
    print('Splitting the data')
    print(f'Test Fraction: {test_fraction}; N folds: {n_folds}')
    print(f'IDs for final testing: {len(ids_for_final_testing)} (total={len(all_uniprot_ids)})')
    print(f'IDs for CV: {len(ids_for_cv)} (total={len(all_uniprot_ids)})')
    print(f'IDs per fold: {ids_per_fold} (total={len(ids_for_cv)})')

    # Shuffle the IDs
    ids_for_cv = np.random.permutation(sorted(ids_for_cv))

    # Split
    folds = [ids_for_cv[ids_per_fold * i + min(i, remainder): ids_per_fold * (i+1) + min(i+1, remainder)] for i in range(n_folds)]

    # Assert that folds are indeed unique
    for i, fold_1 in enumerate(folds):
        for j, fold_2 in enumerate(folds):
            if i != j:
                assert not(set(fold_1) & set(fold_2)), 'folds are overlapping - data leakage!'


    # Adding a special column to the database indicating the fold (and augmented fold)
    current_database['Fold'] = -1
    current_database['Fold Augmented'] = -1
    augmented_folds = {}

    # Iterate through folds + final testing
    for i, fold in tqdm(enumerate(folds + [ids_for_final_testing]), desc='Creating augmented folds...', total=n_folds+1):

        augmented_folds.update({i: []})

        for uniprot in fold:
            uniprot_dict = {'Uniprot ID': uniprot}
            
            # Extract all entries sharing the same UniProt ID
            with_same_uniprot = current_database.loc[current_database["Uniprot ID"] == uniprot, "scPDB ID"].to_list()

            # Choose the first one as main, and the rest as augmenting data
            uniprot_dict.update({'Main': with_same_uniprot[0]})
            uniprot_dict.update({'scPDB IDs': with_same_uniprot})
            uniprot_dict.update({'Count': len(with_same_uniprot)})

            # Update the columns Fold and Fold Augmented in the database
            current_database.loc[current_database["scPDB ID"] == with_same_uniprot[0], "Fold"] = i
            for entry in with_same_uniprot:
                current_database.loc[current_database["scPDB ID"] == entry, "Fold Augmented"] = i

            augmented_folds[i].append(uniprot_dict)

    # Report
    print('\nFolds:')
    for i, fold in augmented_folds.items():

        total_count = sum([line['Count'] for line in fold])
        fold_df = pd.DataFrame(fold)

        if i < 10:
            print(f'\tFold {i}: Main entries: {len(fold)}, Augmented entries: {total_count}')
            fold_df.to_csv(os.path.join(folds_dir, f'{i}.csv'), index=False)
        else:
            print(f'\tFinal Test: Main entries: {len(fold)}, Augmented entries: {total_count}')
            fold_df.to_csv(os.path.join(folds_dir, 'test.csv'), index=False)
    
    current_database.to_csv(database_path, sep='\t')
    return current_database


def get_atoms_properties(database_path: str,
                         exist_ok: bool = True,
                         scpdb_dir: str = 'Data/scPDB',
                         atoms_dir: str = 'Data/Atoms') -> pd.DataFrame:
    """
    Allows to take scPDB entries from the database (loaded from database_path) and
    to read their protein.mol2 and site.mol2 files (from scpdb_dir).

    The structures are processed to create CSV files with atoms with the following columns:
    - ID: atom ID
    - Element: main element symbol
    - Detail: additional details from SYBYL
    - X: X coordinate
    - Y: Y coordinate
    - Z: Z coordinate
    - Charge: charge
    - Neighbour: only for hydrogen atoms

    Plus, chemical binary channels:
    - Positive Ionizable
    - Negative Ionizable
    - Hydrophobic
    - Aromatic
    - Donor
    - Acceptor
    - Metal

    Args:
        database_path (str): _description_
        scpdb_dir (str, optional): _description_. Defaults to 'Data/scPDB'.
        atoms_dir (str, optional): _description_. Defaults to 'Data/Atoms'.

    Returns:
        pd.DataFrame: modified current database
    """

    # Check the input
    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'
    assert isinstance(scpdb_dir, str), f'scpdb_dir must be str, not {type(scpdb_dir)}'
    assert isinstance(atoms_dir, str), f'atoms_dir must be str, not {type(atoms_dir)}'
    os.makedirs(atoms_dir, exist_ok=True)
    metals = {'K', 'Ca', 'Mg', 'Mn', 'Cu', 'Zn', 'Na', 'Li', 'Co'}

    # Read current database
    current_database = pd.read_csv(database_path, sep='\t')
    current_database = current_database.drop(columns=[col for col in current_database.columns if 'unnamed' in col.lower()])

    # Report
    print('Preprocessing of Mol2 files')
    print(f'Total scPDB entries: {len(current_database)}')
    print(f'Expected size of CSVs: {format_size(len(current_database) * 450_000)}')
    skipped_existing = 0
    true_size = 0
    
    for scpdb_id in tqdm(current_database['scPDB ID'], desc='Preprocessing mol2 files...'):

        if os.path.isfile(os.path.join(atoms_dir, scpdb_id + '.csv')) and exist_ok:
            skipped_existing += 1
            continue

        # Read protein.mol2 and site.mol2 files
        protein_file = open(os.path.join(scpdb_dir, scpdb_id, 'protein.mol2')).read()
        site_file = open(os.path.join(scpdb_dir, scpdb_id, 'site.mol2')).read()

        # Get Atoms and Bonds
        protein_atoms, protein_bonds = protein_file.split('@')[2], protein_file.split('@')[3]
        site_atoms, site_bonds = site_file.split('@')[2], site_file.split('@')[3]

        # Identify Hydrogen Neighbours
        oxygen_ids = set(re.findall(r'\n\s{1,7}([0-9]{1,6})\s{1,3}O', protein_atoms))
        nitrogen_ids = set(re.findall(r'\n\s{1,7}([0-9]{1,6})\s{1,3}N', protein_atoms))
        sulphur_ids = set(re.findall(r'\n\s{1,7}([0-9]{1,6})\s{1,3}S', protein_atoms))
        carbon_ids = set(re.findall(r'\n\s{1,7}([0-9]{1,6})\s{1,3}C', protein_atoms))
        hydrogen_ids = set(re.findall(r'\n\s{1,7}([0-9]{1,6})\s{1,3}H', protein_atoms))

        # Here we want to get the partners of hydrogen atoms
        neighbours = {}
        donors = set()
        protein_bonds = protein_bonds.split('\n')[1:]
        for bond in protein_bonds:
            if not bond:
                continue
            
            # Parse the bond
            bond = list(filter(lambda x: x, bond.split(' ')))
            first_atom, second_atom, multiplicity = bond[1], bond[2], bond[3]

            # If multiple bond - skip (cannot be hydrogen)
            if multiplicity != '1':
                continue
            
            # Update donors set and neighbours dict
            if first_atom in hydrogen_ids:

                if second_atom in oxygen_ids:
                    donors.add(first_atom)
                    neighbours.update({first_atom: 'O'})
                elif second_atom in nitrogen_ids:
                    donors.add(first_atom)
                    neighbours.update({first_atom: 'N'})
                if second_atom in sulphur_ids:
                    donors.add(first_atom)
                    neighbours.update({first_atom: 'S'})
                if second_atom in carbon_ids:
                    neighbours.update({first_atom: 'C'})

            elif second_atom in hydrogen_ids:

                if first_atom in oxygen_ids:
                    donors.add(second_atom)
                    neighbours.update({second_atom: 'O'})
                elif first_atom in nitrogen_ids:
                    donors.add(second_atom)
                    neighbours.update({second_atom: 'N'})
                if first_atom in sulphur_ids:
                    donors.add(second_atom)
                    neighbours.update({second_atom: 'S'})
                if first_atom in carbon_ids:
                    neighbours.update({second_atom: 'C'})


        atoms_df = []
        for atom in protein_atoms.split('\n')[1:]:
            if not atom:
                continue
            
            # Parse the atom line
            atom = list(filter(lambda x: x, atom.split(' ')))

            # Get element and detail through SYBYL column
            sybyl = atom[5].split('.')
            if len(sybyl) > 1:
                element, detail = sybyl
            else:
                element, detail = sybyl[0], ''

            # Get Charge
            charge = float(atom[8])
            
            # Append properties
            atoms_df.append([atom[0],  # ID
                             element,  # Main Element Symbol
                             detail,   # Additional Details from SYBYL
                             atom[2],  # X
                             atom[3],  # Y
                             atom[4],  # Z,
                             charge,                                       # charge and basic information
                             neighbours.get(atom[0], None),                # neighbour (only for hydrogens)
                             1 if charge > 0 else 0,                       # Positive Charge Channel
                             1 if charge < 0 else 0,                       # Negative Charge Channel
                             1 if element == 'C' or detail == 'ar' else 0, # Hydrophobic Channel
                             1 if detail == 'ar' else 0,                   # Aromatic Channel
                             1 if atom[0] in donors else 0,                # H-Donor Channel
                             1 if element in {'S', 'O', 'N'} else 0,       # H-Acceptor Channel
                             1 if element in metals else 0]                # Metal Channel
                                    )
        
        # Create dataframe and store it in atoms_dir
        atoms_df = pd.DataFrame(atoms_df, columns=['ID', 'Element', 'Detail', 'X', 'Y', 'Z', 'Charge', 'Neighbour',
                                                    'Positive Ionizable', 'Negative Ionizable', 'Hydrophobic', 'Aromatic', 
                                                    'Donor', 'Acceptor', 'Metal'])
        atoms_df.set_index('ID', inplace=True)
        atoms_df[['X', 'Y', 'Z']] = atoms_df[['X', 'Y', 'Z']].astype('float32')
        path = os.path.join(atoms_dir, scpdb_id + '.csv')
        atoms_df.to_csv(path, sep='\t')
        true_size += os.path.getsize(path)

        # Now we need to find the center of the binding site and store in in the database
        centroid = np.zeros(3)
        n = 0
        for atom in site_atoms.split('\n')[1:]:
            if not atom:
                continue
            n += 1
            atom = list(filter(lambda x: x, atom.split(' ')))
            atom = np.array([float(value) for value in atom[2:5]])
            centroid += atom
        centroid = (np.array(centroid) / n).astype('float32')
        current_database.loc[current_database["scPDB ID"] == scpdb_id, ['Site Center X', 'Site Center Y', 'Site Center Z']] = centroid

    print(f'Total size of actually generated CSVs: {format_size(true_size)}')
    print(f'Skipped as existing: {skipped_existing}')
    current_database.to_csv(database_path, sep='\t')

    return current_database


def visualize_channels(scpdb_id: str,
                       channels: Collection[str] = {'Positive Ionizable', 'Negative Ionizable', 'Hydrophobic', 'Aromatic', 'Donor', 'Acceptor', 'Metal'},
                       atoms_alpha: float = 0.1,
                       show_binding_site: bool = True,
                       database_path: str = 'Data/database.csv',
                       atoms_dir: str = 'Data/Atoms',
                       scpdb_dir: str = 'Data/scPDB',
                       dpi: int = 300):
    
    # Check the input
    assert isinstance(database_path, str), f'database_path must be str, not {type(database_path)}'
    assert isinstance(scpdb_id, str), f'scpdb_id must be str, not {type(scpdb_id)}'
    assert isinstance(atoms_alpha, float | int), f'atoms_alpha must be float or int, not {type(atoms_alpha)}'
    assert isinstance(dpi, int), f'dpi must be int, not {type(dpi)}'
    assert isinstance(atoms_dir, str), f'atoms_dir must be str, not {type(atoms_dir)}'
    assert isinstance(scpdb_dir, str), f'scpdb_dir must be str, not {type(scpdb_dir)}'
    assert isinstance(show_binding_site, bool), f'show_binding_site must be bool, not {type(show_binding_site)}'
    assert isinstance(channels, Collection), f'channels must be Collection, not {type(channels)}'
    possible_channels = {'Positive Ionizable', 'Negative Ionizable', 'Hydrophobic', 'Aromatic', 'Donor', 'Acceptor', 'Metal'}
    for channel in channels:
        assert channel in possible_channels, f'unknown channel "{channel}", consider: {possible_channels}'
    channels = sorted(channels)
    channel_colours = {'Positive Ionizable': 'blue', 'Negative Ionizable': 'red', 'Hydrophobic': 'grey', 'Aromatic': 'darkgrey', 'Donor': 'lightblue', 'Acceptor': 'pink', 'Metal': 'orange'}

    vdw_radii = {'H': 1.1, 'Li': 1.82, 'C': 1.7, 'N': 1.55, 'O': 1.52, 
                'F': 1.47, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.1, 'P': 1.8,
                'S': 1.8, 'Cl': 1.75, 'K': 2.75, 'Ca': 2.31, 'Cu': 1.4,
                'Zn': 1.39, 'Se': 1.9, 'Br': 1.85, 'I': 1.98, 'Na': 2.27,
                'Mn': 2.45, 'Du': 0, 'Fe': 2.44}

    # Read current database
    current_database = pd.read_csv(database_path, sep='\t')
    current_database = current_database.drop(columns=[col for col in current_database.columns if 'unnamed' in col.lower()])

    # Read atoms csv
    atoms_csv = pd.read_csv(os.path.join(atoms_dir, scpdb_id + '.csv'), sep='\t')

    # Load binding site and merging it with atoms_csv to get a "Site" table
    if show_binding_site:
        site = open(os.path.join(scpdb_dir, scpdb_id, 'site.mol2'), 'r').read().split('@')[2]
        site_atoms = []
        for atom in site.split('\n')[1:-1]:
            x, y, z = list(filter(lambda x: x, atom.split(' ')))[2:5]
            site_atoms.append([float(x), float(y), float(z)])
        site_atoms = pd.DataFrame(site_atoms, columns=['X', 'Y', 'Z'])
        site_atoms['Site'] = True
        atoms_csv = pd.merge(atoms_csv, site_atoms, how='left')

    # Plot
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    # The following block is needed to make atom sizes look correct
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width_inch = bbox.width
    xrange = atoms_csv['X'].max() - atoms_csv['X'].min() + 10
    points_per_angstrom = width_inch * 72 / (2 * xrange)

    # plot every atom according to its properties, if show_binding_site, then also add borders to 
    # atoms, that are parts of a binding site
    for i, row in atoms_csv.iterrows():

        ax.scatter(row['X'], row['Y'], row['Z'], marker='o', facecolor='lightgreen', alpha=atoms_alpha,
                    s = (vdw_radii[row['Element']] * points_per_angstrom) ** 2)

        for channel in channels:
            if row[channel] == 1:
                ax.scatter(row['X'], row['Y'], row['Z'], marker='o', color=channel_colours[channel],
                s = (vdw_radii[row['Element']] * points_per_angstrom) ** 2,
                edgecolors = None if show_binding_site and row['Site'] != True else 'black',
                linewidths = 0 if show_binding_site and row['Site'] != True else 0.3)

    # Add the legend with channel colours
    legend = [
        Patch(facecolor=color, edgecolor='k', label=name)
        for name, color in channel_colours.items() if name in channels
    ]
    ax.legend(handles=legend,
            title="Chemical Channels\n(border indicates that atom\nbelongs to the binding site)",
            loc="upper left",
            frameon=True,
            fontsize=int(dpi/30),
            title_fontsize=int(dpi/25))

    # Set reasinable limits
    ax.set_xlim(atoms_csv['X'].min() - 5, atoms_csv['X'].max() + 5)
    ax.set_ylim(atoms_csv['Y'].min() - 5, atoms_csv['Y'].max() + 5)
    ax.set_zlim(atoms_csv['Z'].min() - 5, atoms_csv['Z'].max() + 5)

    # Turn off the grid and add a title
    ax.grid(False)
    ax.set_title(f'Entry {scpdb_id}, channels shown: {", ".join(channels)}', fontsize=int(dpi/20))
    plt.show()
