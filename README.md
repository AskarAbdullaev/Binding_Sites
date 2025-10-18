# Binding_Sites
Practical Work on Binding Sites (bases on DeepSite approach)

## Dependencies

For this project I used the following modules:

| Module         | Version |
| -------------- | ------- |
| Python version | 3.12.2  |
| requests       | 2.32.2  |
| pandas         | 2.2.2   |
| numpy          | 1.26.4  |
| regex          | 2.5.135 |
| tqdm           | 4.66.4  |
| bs4            | 4.12.3  |
| matplotlib     | 3.8.4   |
| torch          | 2.4.1   |
| sklearn        | 1.4.2   |

## File structure

```text
./
|
|- all_scpdb_entries.txt (list of all scPDB entries as of Sep.'25)
|- excluded_scpdb_entries.txt (entries that are filtered out during preprocessing)
|- decode.npy (decoder of atom properties)
|- orogonal_split.txt (data split provided in DeepSite supplementary files)
|
|- data_utilities.py (utilities for the 'load_and_process' notebook)
|- load_and_process.ipynb (notebook for data loading and preprocessing)
|
|- sample_and_train_utilities.py (utilities for sample_and_train' notebook)
|- sample_and_train.ipynb (notebook for voxelization, sampling and hyperparameters search)
|
|- Data 
    |
    |- scPDB (original scPDB dataset)
        |- 1a2b_1
            |- protein.mol2
            |- site.mol2
            |- ...(other possible files from scPDB)
        |- ...(all scPDB entries)
    |
    |- SCOPe (folder with SCOPe versions)
        |- 2_08.csv
        |- ...(possibly other SCOPe versions)
    |
    |- Pages (textual files with scPDB web-pages source codes)
        |- 1a2b_1.txt
        |- ...(all the source codes of scPDB web pages)
    |
    |- Folds (CSVs with data split / folds)
        |- 1.csv
        |- ...(other folds)
        |- test.csv
    |
    |- Atoms (CSVs with processed atoms, including chemical channels)
        |- 1a2b_1.csv
        |- ...(csvs with atom coords and props)
    |
    |- Voxels (voxelized entries in compact form: atom_grid + occupancy + decoder (common))
        |- 1
            |- 1a2b_1
                |- atoms_grid.npy
                |- occupancy.npy
                |- site_center.npy
            |- ... (other voxelized entries with voxel size 1)
        |
        |- 2
            |- 1a2b_1
                |- atoms_grid.npy
                |- occupancy.npy
                |- site_center.npy
            |- ... (other voxelized entries with voxel size 2)
    |
    |- CV (Training logs)
        |- pilot (folder for hyperparameter search)
            |- bs_128_do_0_cs_1
                |- train_loss.txt
                |- test_loss.txt
                |- true_labels.npy
                |- predictions.npy
            |- ... (other hyperparameter combinations)
```

## Introduction

Proteins are large complex molecules composed of sequences of 20 different amino acids
arranged in one or more polypeptide chains. Their three-dimensional structures form
intricate surfaces with numerous concavities and protrusions, creating distinct microen-
vironments for ligand binding and catalysis [23]. These regions are commonly referred to
as (ligand-)binding pockets, (ligand-)binding sites.

The identification and characterization of druggable pockets play a crucial role in in
silico target-based drug discovery, a critical step in structure-based drug design [34].
However, traditional drug discovery methods are often time-consuming and expensive
[33]. Thus, nn recent years, computational approaches have been increasingly adopted
by the industry to improve the efficiency and effectiveness of the drug discovery process
[33]. Noticeably, deep learning has become a powerful tool in computational biology [2].

In 2017 article DeepSite [17] proposed an algorithm theh predicts binding sites (pockets)
for arbitrary protein structures both at the point-wise level and in 3D space. While ear-
lier tools such as FPocket [20], POCKET [22], Pocket-Picker [32], SURFNET [19], and
Concavity [5] have been previously developed for this task, they mainly rely on geomet-
ric analyses (using, for example, alpha-spheres [20] or probing spheres between protein
atoms [22, 32]) or combine geometric data with evolutionary conservation [5]. In con-
trast, DeepSite leverages a 3D deep convolutional neural network (DCNN) architecture
and incorporates the chemical features of individual atoms.

A convolutional neural network (CNN) is a type of neural network specifically designed
to process grid-like data structures such as images or volumetric data [14]. The main
principle of CNNs is similar to traditional neural networks: the input is transformed
through multiple layers, where each layer applies a weighted transformation (optionally
with a bias term), followed by a non-linear activation function. These weights and biases
are learnable parameters updated during training [14]. CNNs benefit from the local prop-
erties of the input through the use of convolutional layers, where neurons are connected
only to small, localized regions of the input and share weights across the input space.
This significantly reduces the number of parameters and improves the performance.
By stacking many different layers, the DCNN (deep convolutional neural network) archi-
tecture is obtained. In the context of 3D protein data, DCNNs include 3D convolutional
layers that operate over three spatial dimensions and use 3D filters (kernels). As it is
shown by [30], 3D convolutional neural networks are able to surpass traditional docking
algorithms when applied to proteins.

The main goal of this research is to apply the similar approach as suggested by DeepSite,
while using the current version of the scPDB database [10] in order to estimate if the
success of DeepSite is reproducible and scalable. Due to computational limitations, the
results will be compared not only to DeepSite and FPocket, but also to baseline classifiers:
random classifier and a decision tree classifier. The additional objectives of this research
are to estimate the influence of voxel size, number of channels and distance measure on
the final result, which was not disussed in the DeepSite paper.
This research is meant to contribute to the reproducibility and reassessment of traditional
deep learning methods in structural bioinformatics. Moreover, investigation of hyperpa-
rameters may reveal additional insights for future development of the method.


## Materials

The primary source of protein structures and related information for the study is scPDB
[10], an annotated database of druggable binding sites derived from the Protein Data
Bank (PDB) [3]. The scPDB mainly includes information on small synthetic and natural
ligands, along with corresponding high-quality, non-redundant protein binding sites. For
more structural information and for later analyses, I will also make use of the latest stable
release of a SCOPe database [6]: a database, which classifies protein chains according to
their structure, function and taxonomy.

By the 14th October 2025, there are 17,594 entries available for download from the official
page of scPDB. However, according to the information on the same site, the last stable
release (2017) contains 16034 entries, 4782 proteins and 6326 ligands. To avoid unchecked
entries, erroneous entries and duplicates in the dataset, the thorough parsing of individual
web-pages is performed. Using the BeautifulSoup4 module [29] I have successfully found
16033 entries, which is only one less than a stable release contains. Also, additional
valuable information about the entries is obtained, some of which is summarized in the
Table 1

According to the aggregated table:

1. PDB IDs are not unique across entries, 783 entries have the IDS already present in
the database;
2. Unique UniProts [9] (unique proteins themselves) are much less numerous - only
about 4690, which makes the database structurally redundant.
3. Mean resolution is more then 2Å.
4. Majority of structures come from Eukaryotic organisms and around a third - from
Homo sapiens.
5. Although there are hundreds of SCOPe families, several are largely over-represented
in the database
6. About 20% of binding sites contain metal atoms (mostly Mg, Zn or Mn) and only
9% contain co-factors (mostly NAP, FAD or NDP)
7. There on average 36 residues per binding site, with mean volume being around 800 Å³
8. Cavities have Hydrophobic and Polar regions on average in the same proportion.

Ligand properties are not of much interest to us in the scope of this paper.

To prevent erroneous evaluation, only a single example featuring the distinct protein
structure (unique UniProt ID). 25% of samples (1172) are held-out for the final test set
to compute the domain specific-metrics.

The remaining dataset is evaluated using the 5-fold cross-validation [19] approach (ini-
tially, 10-fold procedure was chosen but abandoned due to computational reasons). In
each fold, 2811 - 2814 samples were used for training and 702 - 704 for testing. Accord-
ing to [4], cross-validation is particularly suitable when a disjoint validation set is not
feasible.


Table 1: Selected aggregated values of scPDB (v.2017):
| Metric                                          | Value(s)                                                             |
|:------------------------------------------------|:---------------------------------------------------------------------|
| Total Entries                                   | 16033                                                                |
| Unique PDB IDs                                  | 15250                                                                |
| Unique Uniprot IDs                              | 4690                                                                 |
| Mean Resolution (Å)                             | 2.14                                                                 |
| Number of Unique Species                        | 927                                                                  |
| Most Common Species                             | Homo sapiens (5406), Escherichia coli (977), Rattus norvegicus (568) |
| Number of Unique Reigns                         | 5                                                                    |
| Most Common Reign                               | Eukaryota (9199), Bacteria (5308), Viruses (949)                     |
| SCOPe classes present                           | 597                                                                  |
| Most Common SCOPe classes                       | c.2.1 (2767), l.1.1 (2676), d.144.1 (2436)                           |
| Binding Sites with Metals                       | 3323                                                                 |
| Most Common Metals in Binding Sites             | MG (2027), ZN (930), MN (367)                                        |
| Binding Sites with Cofactors                    | 1389                                                                 |
| Most Common Cofactors in Binding Sites          | NAP (345), FAD (286), NDP (187)                                      |
| Number of Residues in Binding Site (mean)       | 36.45                                                                |
| Standard Amino Acids in Binding Site (mean)     | 34.49                                                                |
| Non Standard Amino Acids in Binding Site (mean) | 0.49                                                                 |
| Water Molecules in Binding Site (mean)          | 1.47                                                                 |
| Cavity Ligandability (mean)                     | 0.8566253976174142                                                   |
| Cavity Volume (Å³) (mean)                       | 792.66                                                               |
| Cavity % Hydrophobic (mean)                     | 49.06                                                                |
| Cavity % Polar (mean)                           | 50.94                                                                |
| Ligand Molecular Weight (mean)                  | 475.83                                                               |
| Ligand Buried Surface Area (Å²) (mean)          | 63.84                                                                |
| Ligand Polar Surface Area (Å²) (mean)           | 220.15                                                               |
| Ligand H-Bond Acceptors (mean)                  | 10.45                                                                |
| Ligand H-Bond Donors (mean)                     | 3.50                                                                 |
| Ligand Rings (mean)                             | 3.41                                                                 |
| Ligand Aromatic Rings (mean)                    | 2.00                                                                 |
| Ligand Anionic Atoms (mean)                     | 1.51                                                                 |
| Ligand Cationic Atoms (mean)                    | 0.35                                                                 |
| Ligand Rule of Five Violation (mean)            | 1.12                                                                 |
| Ligand Rotatable Bonds (mean)                   | 7.83                                                                 |


## Data Preprocessing

Table 2: Rules (atom types) used for chemical feature descriptors: DeepSite
[17] / modified for this paper
| Channel  | DeepSite [17] Rule | Modified Rule |
| -------- | ------------------ | ------------- |
| Hydrophobic | atom type C or A | atom types C, C.ar, N.ar |
| Aromatic | atom type A | atom types C.ar, N.ar |
| Hydrogen bond acceptor | atom type NA or NS or OA or OS or SA | atom types N, O, S |
| Hydrogen bond donor | atom type HD or HS with O or N partner | atom type H with N, O, S partner |
| Positive ionizable | atom with positive charge | atom with positive charge | 
| Negative ionizable | atom with negative charge | atom with negative charge |
| Metal | atom type MG or ZN or MN or CA or FE | every metal atom |
| Excluded volume | all atom types | all atom types except for placeholders |

Each protein structure is treated as a three-dimensional image with a resolution of 1 ×
1 × 1 Å3 or 2 × 2 × 2 Å3 per voxel (as the mean resolution of 3D structures appeared to be around 2Å, it is worth testing both voxel sizes). A voxel (volumetric pixel) is the 3D analog of a 2D pixel, representing a value on a regular three-dimensional grid.

Just as pixels in standard images contain color information for different channels (e.g.,
RGB), each protein voxel contains spatial/chemical information. For this paper 8 channels
are chosen (suggested in the DeepSite): hydrophobic, aromatic, hydrogen bond acceptor,
hydrogen bond donor, positive ionizable, negative ionizable, metal, and excluded volume.
These features are assigned to atoms by analyzing Tripos Mol2 files [32] of proteins - a
widely used format with descriptions of atoms, bonds, and substructures. Although the
DeepSite authors used an external tool to extract atom properties (AutoDock4, [27]),
I use simpler heuristics. The table with AutoDock4 atom types (DeepSite) and Tripos
SYBYL atom types associated with channels can be found in the Table 2.
Folowing the DeepSite idea, I also use a spacial metric to estimate the voxel channel
values - a continuous occupancy score, defined as:

$$
n(r) = 1 - \exp\left(-\left(\frac{r_{\text{vdw}}}{r}\right)^{12}\right)
$$

where $r_{vdw}$ denotes the Van der Waals radius of an atom — i.e., the distance at which
another atom can approach without significant repulsion [28] — and r is the Euclidean
distance from the atom to the voxel center. This function serves as a normalized measure
of the influence of the atom on the voxel, mapping the values to the range [0, 1) by design.
The algorithm computes the occupancy score of each atom with respect to each voxel
center and assigns to the voxel descriptor the chemical properties of the atom with the
highest occupancy score. This results in an 8-channel representation of each voxel based
on the most relevant nearby atom.

The computational burden of a naive computation would have the computational com-
plexity for one protein $O(N · A3/V3)$, where N - the number of atoms per protein; A -
the maximum span of coordinates; V - the chosen voxel size. To decrease the complexity,
I make use of the float32 format, which will be used for the model. The resolution of
torch.float32 datatype is 1.2 · 10−6, thus:

$$
(1 - \epsilon) \leq \exp\left(-\left(\frac{r_{\text{vdw}}}{r}\right)^{12}\right)
$$
$$
r \geq r_{\text{vdw}} / (-ln(1-\epsilon))^{-12} \approx 3.2 \cdot r_{\text{rdw}}
$$


At a distance greater then 3.2 times its own radius an atoms has practically no influence
on the occupancy scores of voxels. However, the majority of protein structures in scPDB
were obtained by X-ray with mean resolution around 2Å. According to Helliwell et al.
[15], the linear uncertainty of such a method is 0.1-0.3Å, which means 0.3Å distance
uncertainty, and I can safely approximate the formula as r ≤ 3 · rvdw. Supposing that
the atom center belongs to a voxel $(x_a, y_a, z_a)$, the maximum effective span would be
$1 + ⌊(3 · r_{vdw} − V/2)/V⌋$ in every direction, where V - voxel size.
The example of original and voxelized layouts for the structure ’2z08_1’ can be seen in
Figure 1.

To limit and standardize the input size of the model and focus on the local structure,
the protein grid is divided into cubic subgrids of size 16 × 16 × 16 $Å^3$ (16 or 8 voxels per
side depending on voxel size). The entire protein grid is padded with an 8 Å margin in
each direction to ensure that each protein voxel appears in the central part of at least
one subgrid.

These subgrids are extracted with a sliding window sampling using a stride of 4 Å. Each
subgrid is labeled as positive if the distance between its geometric center and the an-
notated binding pocket center (from the scPDB database) is less than or equal to 4 Å;
otherwise, the subgrid is labeled as negative. Mathematically, it means that one binding
site produces from 3 to 8 positive subgrids. This enables the model to learn localized
features that distinguish true binding sites from surrounding regions.

Figure 1. Chemical channels highlighted in the atom layout (left) and in the voxel layout (right). Non-highlighted atoms/voxels rendered as semi-transparent green objects. (Structure shown: 2z08\_1)


<img width="495" height="790" alt="atoms_and_voxels" src="https://github.com/user-attachments/assets/d2ac96d2-bdec-459b-a101-437d8ef1417e" />


## Architecture

The model architecture is inspired by the DeepSite architecture. The model takes a 16 ×
16 × 16Å (the exact number of voxels depends on the voxel size) with 8 pharmacophoric
channels as input.

The first convolutional block applies two 3D convolution layers: an 8×8×8Å convolution
followed by a 4×4×4Å convolution. The output is then downsampled using 2×2×2 max
pooling and regularized with a dropout layer. This block is repeated with an updated
configuration: 4 × 4 × 4Å convolution layers. The exact kernel sizes depend on the chosen
voxel sizes to make the receptive fields of the kernels comparable.

The final portion of the model consists of a dense layer and an output neuron with a
sigmoid activation function, which outputs a probability for the binary classification task
(is there a binding site within 4 Å from the subgrid center or not). Each convolutional and
dense layer is followed by an ELU (Exponential Linear Unit) activation, except for the
last output, which is processed using sigmoid to convert it into probability. The number
of parameters for this architecture is moderate (see Table ??).Notice that the number of
parameters is comparable regardless the size of the voxel due to the scaling parameter
’k’ in the model definition. The model is implemented in PyTorch [26]. For the model
architecture Python code see Listing 1.

Table 3: Number of trainable parameters for different model configurations.
| Channel size | Voxel size | Parameters |
| ------------ | ---------- | ---------- |
| ×1 | 1 | 917,921 |
| ×2 | 1 | 1,409,505 |
| ×1 | 2 | 873,472 |
| ×2 | 2 | 1,321,861 |

Listing 1: 3D DCNN model architecture.
```python
self.dropout = dropout
self.channel_size = channel_size
self.voxel_size = voxel_size
self.k = self.voxel_size ** 1.5

self.model = torch.nn.Sequential(
    torch.nn.Conv3d(
        8,
        int(16 * self.channel_size * self.k),
        kernel_size=8 // self.voxel_size,
        padding='same'),
    torch.nn.ELU(),
    torch.nn.Conv3d(
        int(16 * self.channel_size * self.k),
        int(16 * (self.channel_size + 1) * self.k),
        kernel_size=4 // self.voxel_size,
        padding='same'),
    torch.nn.ELU(),
    torch.nn.MaxPool3d(2),
    torch.nn.Dropout3d(dropout),
    torch.nn.Conv3d(
        int(16 * (self.channel_size + 1) * self.k),
        int(16 * (self.channel_size + 2) * self.k),
        kernel_size=4 // self.voxel_size,
        padding='same'),
    torch.nn.ELU(),
    torch.nn.Conv3d(
        int(16 * (self.channel_size + 2) * self.k),
        int(16 * (self.channel_size + 3) * self.k),
        kernel_size=4 // self.voxel_size,
        padding='same'),
    torch.nn.ELU(),
    torch.nn.MaxPool3d(2),
    torch.nn.Dropout3d(dropout),
    torch.nn.Flatten(),
    torch.nn.Linear(
        (4 // self.voxel_size) ** 3 * int(16 * (self.channel_size + 3) * self.k),
        int(128 * self.k)),
    torch.nn.ELU(),
    torch.nn.Dropout(dropout * 2),
    torch.nn.Linear(int(128 * self.k), 1)
)
```

## Hyperparameters Search

Although the model architecture is based on the DeepSite model, the amount of training
data is significantly less in this research. Thus, it is required to perform a search for
new hyperparameters. The computational complexity of the procedure made me limit
the search to the following grid:

- Batch Size: 128 or 256
- Dropout Rate: 0, 0.1 or 0.25
- Channel Size Multiplier: x1 or x2
  
For the search I only use a single fold with the maximum number of epochs equal to
100 and an early stopping triggered after 5 epochs without test loss improvement. The
optimizer is Adam [17], loss function is binary cross-entropy, positive weight is set to 1.
The grid search takes 35 hours on 12-core CPU (2023). The results for each run can be
seen in Figure 2. As visible in the plot, the lowest test loss is reached for the batch size 128, 0 dropout rate
and 16x2 channels multiplier. The epoch at which the best loss is reached is 8. However,
to account for variance in folds, I will set the maximum epochs to 15 with early stopping
triggered after 3 non-improving epochs.

<img width="1189" height="1180" alt="hp_grid" src="https://github.com/user-attachments/assets/9d3adb90-198e-44c2-b151-efb3077b9702" />


## Cross-Validation

As mentioned before, there are 2 model architectures that are trained: Deep 3D-CNN
with voxel sizes of 1Å and 2Å. To estimate the required complexity, I have decided to also
train and evaluate the linear models with the architecture that can be seen in Listing 2.
Such models only have 32769 or 4097 trainable parameters depending on the voxel size
(much less compared to Table 3).
Thus, 4 models in total are trained and evaluated over 5 balanced and unique folds:
1. CNN (voxel size 1), 1409505 parameters
2. CNN (voxel size 2), 1321861 parameters
3. Linear (voxel size 1), 32769 parameters
4. Linear (voxel size 2), 4097 parameters


The training is done using 128 batch size, 12 epochs with patience 3, the Adam optimizer
[16] and BCE loss with logits. The training of all the 4 models takes around 30 hours on
the 12-core CPU (2023).
For every model the train and test losses are summarized using the mean and confidence
intervals per epoch. As it can be observed from Figure 3, the CNN architecture with voxel
size 1Å shows significantly lower test loss values after a proper number of epochs. Also,
it appears that the models using voxel size 2Å performs worse than those with smaller
voxels. Apparently, enlargement of voxels leads to the loss of 82,5% of spati0-chemical
information, which negatively affects the performance.
Although the superiority of CNN (1Å) is already clear, let’s also take a look at the ROC
AUC scores and Average Precision scores of the 4 models: Figure 4, Figure 5. Surprisingly,
the relatively simple Linear model (1Å) shows around 90% AUC score, while containing
40 times less parameters than 3D CNN (1Å). The summary of model evaluations can be
found in in Table 4.

Figure 3. Loss per epoch for every trained model. Dashed lines indicate the
best mean loss. Red dashed lines - the best over all the models. The
semi-transparent margins indicate the confidence interval (p=0.05)
<img width="790" height="789" alt="cv_losses" src="https://github.com/user-attachments/assets/37cb6d4c-60a7-4d63-af65-775fe789cc65" />

AUC scores of the 4 models (evaluated at the best epoch by test
loss) with confidence interval margins (p=0.05)
<img width="1408" height="858" alt="cv_auc" src="https://github.com/user-attachments/assets/dfd0a8d4-a55d-4063-b7cd-14f9affa0efe" />


Figure 5: Average Precision scores of the 4 models (evaluated at the best
epoch by test loss) with confidence interval margins (p=0.05)
<img width="1408" height="858" alt="cv_aps" src="https://github.com/user-attachments/assets/8cb154ff-35c5-4167-a384-6da589c917f4" />

Table 4: Evaluation metrics of models based on 5-fold cross-validation. All
confidence intervals are computed using p=.05, random baseline as-
sumes balanced dataset.
| Model        |   Voxel Size |  Best Epoch (avg) |  Best Loss (avg) | AP                | ROC AUC           |   Accuracy      |  F1       |
|:-------------|-------------:|------------------:|-----------------:|:------------------|:------------------|----------------:|----------:|
| cnn          |            1 |                 8 |         0.0810   | 0.9945 ± 0.0017   | 0.9949 ± 0.0012   |        0.9722   |  0.9743   |
| cnn          |            2 |                 3 |         0.6396   | 0.6653 ± 0.0467   | 0.6961 ± 0.0527   |        0.6301   |  0.6728   |
| linear       |            1 |                 5 |         0.4540   | 0.9277 ± 0.0052   | 0.9105 ± 0.0075   |        0.8193   |  0.8468   |
| linear       |            2 |                 4 |         0.6433   | 0.6770 ± 0.0211   | 0.7103 ± 0.0222   |        0.6511   |  0.7184   |
| random       |            - |                 - |         0.6930   | 0.500             | 0.500             |        0.5      |  0.5      |

Based on the cross-validation result, the bast model (CNN (1\text{\AA}) is then trained on the complete CV-subset to obtain the final model. The linear model (1\text{\AA}) is also trained on the complete CV-subset (as it showed surprisingly high evaluation results, it is worth keeping it for further stages).

## Domain-Specific Results

TBA
(As I have the separate test set, not intersecting with CV data, I will test the resulting model with DCC and DVO metrics. When ready, I can compare it to DeepSite results and FPocket results)

## Conclusion and Discussion

TBA

## References

[1] Acellera. 2025. PlayMolecule: Molecular Discovery Platform. Accessed: 2025-03-28.
https://www.playmolecule.org.

[2] Christof Angermueller, Tanel Pärnamaa, Leopold Parts, and Oliver Stegle. 2016.
Deep learning for computational biology. en. Molecular Systems Biology, 12, 7,
(July 2016), 878. issn: 1744-4292, 1744-4292. doi: 10.15252/msb.20156651. Re-
trieved 03/28/2025 from https://www.embopress.org/doi/10.15252/msb.2015665
1.

[3] H. M. Berman. 2000. The Protein Data Bank. Nucleic Acids Research, 28, 1, (Jan-
uary 2000), 235–242. issn: 13624962. doi: 10 . 1093 / nar / 28 . 1 . 235. Retrieved
03/28/2025 from https : / / academic . oup . com / nar / article - lookup / doi / 10 . 1093
/nar/28.1.235.

[4] Daniel Berrar. 2019. Cross-Validation. en. In Encyclopedia of Bioinformatics and
Computational Biology. Elsevier, pp. 542–545. isbn: 9780128114322. doi: 10.1016
/B978-0-12-809633-8.20349-X. Retrieved 03/28/2025 from https://linkinghub.els
evier.com/retrieve/pii/B978012809633820349X.

[5] John A. Capra, Roman A. Laskowski, Janet M. Thornton, Mona Singh, and
Thomas A. Funkhouser. 2009. Predicting Protein Ligand Binding Sites by Com-
bining Evolutionary Sequence Conservation and 3D Structure. en. PLoS Compu-
tational Biology, 5, 12, (December 2009), e1000585. Thomas Lengauer, (Ed.) issn:
1553-7358. doi: 10.1371/journal.pcbi.1000585. Retrieved 03/28/2025 from https:
//dx.plos.org/10.1371/journal.pcbi.1000585.

[6] John-Marc Chandonia, Lindsey Guan, Shiangyi Lin, Changhua Yu, Naomi K Fox,
and Steven E Brenner. 2022. SCOPe: improvements to the structural classification
of proteins – extended database to facilitate variant interpretation and machine
learning. en. Nucleic Acids Research, 50, D1, (January 2022), D553–D559. issn:
0305-1048, 1362-4962. doi: 10 . 1093 / nar / gkab1054. Retrieved 04/04/2025 from
https://academic.oup.com/nar/article/50/D1/D553/6447236.

[7] François Chollet et al. 2015. Keras. https://keras.io. (2015).

[8] D. Comaniciu and P. Meer. 2002. Mean shift: a robust approach toward feature
space analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence,
24, 5, (May 2002), 603–619. issn: 01628828. doi: 10.1109/34.1000236. Retrieved
03/28/2025 from http://ieeexplore.ieee.org/document/1000236/.

[9] Jérémy Desaphy, Karima Azdimousa, Esther Kellenberger, and Didier Rognan.
2012. Comparison and Druggability Prediction of Protein–Ligand Binding Sites
from Pharmacophore-Annotated Cavity Shapes. en. Journal of Chemical Informa-
tion and Modeling, 52, 8, (August 2012), 2287–2299. issn: 1549-9596, 1549-960X.
doi: 10.1021/ci300184x. Retrieved 03/28/2025 from https://pubs.acs.org/doi/10
.1021/ci300184x.

[10] Jérémy Desaphy, Guillaume Bret, Didier Rognan, and Esther Kellenberger. 2015.
sc-PDB: a 3D-database of ligandable binding sites—10 years on. en. Nucleic Acids
Research, 43, D1, (January 2015), D399–D404. issn: 1362-4962, 0305-1048. doi:
10.1093/nar/gku928. Retrieved 03/28/2025 from http://academic.oup.com/nar/a
rticle/43/D1/D399/2439494/scPDB-a-3Ddatabase-of-ligandable-binding-sites10.

[11] S. Doerr, M. J. Harvey, Frank Noé, and G. De Fabritiis. 2016. HTMD: High-
Throughput Molecular Dynamics for Molecular Discovery. en. Journal of Chemical
Theory and Computation, 12, 4, (April 2016), 1845–1852. issn: 1549-9618, 1549-
9626. doi: 10.1021/acs.jctc.6b00049. Retrieved 04/04/2025 from https://pubs.acs
.org/doi/10.1021/acs.jctc.6b00049.

[12] Naomi K. Fox, Steven E. Brenner, and John-Marc Chandonia. 2014. SCOPe: Struc-
tural Classification of Proteins—extended, integrating SCOP and ASTRAL data
and classification of new structures. en. Nucleic Acids Research, 42, D1, (January
2014), D304–D309. issn: 0305-1048, 1362-4962. doi: 10 . 1093 / nar / gkt1240. Re-
trieved 04/04/2025 from https://academic.oup.com/nar/article-lookup/doi/10.10
93/nar/gkt1240.

[13] Naomi K. Fox, Steven E. Brenner, and John-Marc Chandonia. 2014. SCOPe: Struc-
tural Classification of Proteins—extended, integrating SCOP and ASTRAL data
and classification of new structures. en. Nucleic Acids Research, 42, D1, (January
2014), D304–D309. issn: 0305-1048, 1362-4962. doi: 10 . 1093 / nar / gkt1240. Re-
trieved 03/28/2025 from https://academic.oup.com/nar/article-lookup/doi/10.10
93/nar/gkt1240.

[14] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. http:
//www.deeplearningbook.org. MIT Press.

[15] 2013. Computer Graphics: Principles and Practice. (3rd ed.). Addison-Wesley,
Boston. Chapter 14, pp. 349–350. isbn: 978-0-321-39952-6.

[16] Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J.
Dally, and Kurt Keutzer. 2016. SqueezeNet: AlexNet-level accuracy with 50x fewer
parameters and &lt;0.5MB model size. (2016). doi: 10.48550/ARXIV.1602.07360.
Retrieved 03/28/2025 from https://arxiv.org/abs/1602.07360.

[17] J Jiménez, S Doerr, G Martínez-Rosell, A S Rose, and G De Fabritiis. 2017. Deep-
Site: protein-binding site predictor using 3D-convolutional neural networks. en.
Bioinformatics, 33, 19, (October 2017), 3036–3042. Alfonso Valencia, (Ed.) issn:
1367-4803, 1367-4811. doi: 10.1093/bioinformatics/btx350. Retrieved 03/28/2025
from https://academic.oup.com/bioinformatics/article/33/19/3036/3859178.

[18] Diederik P. Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Opti-
mization. (2014). doi: 10.48550/ARXIV.1412.6980. Retrieved 04/04/2025 from
https://arxiv.org/abs/1412.6980.

[19] Roman A. Laskowski. 1995. SURFNET: A program for visualizing molecular sur-
faces, cavities, and intermolecular interactions. en. Journal of Molecular Graphics,
13, 5, (October 1995), 323–330. issn: 02637855. doi: 10.1016/0263-7855(95)00073
-9. Retrieved 03/28/2025 from https://linkinghub.elsevier.com/retrieve/pii/02637
85595000739.

[20] Vincent Le Guilloux, Peter Schmidtke, and Pierre Tuffery. 2009. Fpocket: An open
source platform for ligand pocket detection. en. BMC Bioinformatics, 10, 1, (De-
cember 2009), 168. issn: 1471-2105. doi: 10.1186/1471- 2105- 10- 168. Retrieved
03/28/2025 from https://bmcbioinformatics.biomedcentral.com/articles/10.1186
/1471-2105-10-168.

[21] Johannes Lederer. 2021. Activation Functions in Artificial Neural Networks: A
Systematic Overview. (2021). doi: 10 . 48550 / ARXIV . 2101 . 09957. Retrieved
03/28/2025 from https://arxiv.org/abs/2101.09957.

[22] David G. Levitt and Leonard J. Banaszak. 1992. POCKET: A computer graphies
method for identifying and displaying protein cavities and their surrounding amino
acids. en. Journal of Molecular Graphics, 10, 4, (December 1992), 229–234. issn:
02637855. doi: 10.1016/0263-7855(92)80074-N. Retrieved 03/28/2025 from https:
//linkinghub.elsevier.com/retrieve/pii/026378559280074N.

[23] Jie Liang, Clare Woodward, and Herbert Edelsbrunner. 1998. Anatomy of protein
pockets and cavities: Measurement of binding site geometry and implications for
ligand design. en. Protein Science, 7, 9, (September 1998), 1884–1897. issn: 0961-
8368, 1469-896X. doi: 10.1002/pro.5560070905. Retrieved 03/28/2025 from https:
//onlinelibrary.wiley.com/doi/10.1002/pro.5560070905.

[24] Rushi Longadge and Snehalata Dongre. 2013. Class Imbalance Problem in Data
Mining Review. (2013). doi: 10.48550/ARXIV.1305.1707. Retrieved 03/28/2025
from https://arxiv.org/abs/1305.1707.

[25] H. B. Mann and D. R. Whitney. 1947. On a Test of Whether one of Two Random
Variables is Stochastically Larger than the Other. en. The Annals of Mathematical
Statistics, 18, 1, (March 1947), 50–60. issn: 0003-4851. doi: 10.1214/aoms/1177730
491. Retrieved 04/04/2025 from http://projecteuclid.org/euclid.aoms/1177730491.

[26] Garrett M. Morris, Ruth Huey, William Lindstrom, Michel F. Sanner, Richard K.
Belew, David S. Goodsell, and Arthur J. Olson. 2009. AutoDock4 and AutoDock-
Tools4: Automated docking with selective receptor flexibility. en. Journal of Com-
putational Chemistry, 30, 16, (December 2009), 2785–2791. issn: 0192-8651, 1096-
987X. doi: 10.1002/jcc.21256. Retrieved 03/28/2025 from https://onlinelibrary.w
iley.com/doi/10.1002/jcc.21256.

[27] Linus Pauling. 2010. The nature of the chemical bond and the structure of molecules
and crystals: an introduction to modern structural chemistry. eng. (3. ed., 17.
print ed.). Cornell Univ. Press, Ithaca, NY. isbn: 9780801403330.

[28] Karen Simonyan and Andrew Zisserman. 2014. Very Deep Convolutional Networks
for Large-Scale Image Recognition. (2014). doi: 10 . 48550 / ARXIV . 1409 . 1556.
Retrieved 03/28/2025 from https://arxiv.org/abs/1409.1556.

[29] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alex Alemi. 2016.
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learn-
ing. (2016). doi: 10.48550/ARXIV.1602.07261. Retrieved 03/28/2025 from https:
//arxiv.org/abs/1602.07261.

[30] Izhar Wallach, Michael Dzamba, and Abraham Heifets. 2015. AtomNet: A Deep
Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug
Discovery. (2015). doi: 10.48550/ARXIV.1510.02855. Retrieved 03/28/2025 from
https://arxiv.org/abs/1510.02855.

[31] Xipeng Wang, Simón Ramírez-Hinestrosa, Jure Dobnikar, and Daan Frenkel. 2020.
The Lennard-Jones potential: when (not) to use it. en. Physical Chemistry Chemical
Physics, 22, 19, 10624–10633. issn: 1463-9076, 1463-9084. doi: 10.1039/C9CP054
45F. Retrieved 03/28/2025 from https://xlink.rsc.org/?DOI=C9CP05445F.

[32] Martin Weisel, Ewgenij Proschak, and Gisbert Schneider. 2007. PocketPicker: anal-
ysis of ligand binding-sites with shape descriptors. en. Chemistry Central Journal,
1, 1, (December 2007), 7. issn: 1752-153X. doi: 10.1186/1752-153X-1-7. Retrieved
03/28/2025 from https://bmcchem.biomedcentral.com/articles/10.1186/1752-153

[33] Yue Zhang, Mengqi Luo, Peng Wu, Song Wu, Tzong-Yi Lee, and Chen Bai. 2022.
Application of Computational Biology and Artificial Intelligence in Drug Design.
en. International Journal of Molecular Sciences, 23, 21, (November 2022), 13568.
issn: 1422-0067. doi: 10.3390/ijms232113568. Retrieved 03/28/2025 from https:
//www.mdpi.com/1422-0067/23/21/13568.

[34] Xiliang Zheng, LinFeng Gan, Erkang Wang, and Jin Wang. 2013. Pocket-Based
Drug Design: Exploring Pocket Space. en. The AAPS Journal, 15, 1, (January
2013), 228–241. issn: 1550-7416. doi: 10 . 1208 / s12248 - 012 - 9426 - 6. Retrieved
03/28/2025 from http://link.springer.com/10.1208/s12248-012-9426-6.

