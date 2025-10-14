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


## References

References
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

