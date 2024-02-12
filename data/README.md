Some files are too large to be pushed to GitHub, therefore, all the files are provided via google drive: https://drive.google.com/drive/folders/17z34rgAw2nz4ywlF2G4CTJifcAI66lka?usp=sharing

Folder aggregation - contains well-established dataset for aggregation prediction model training and testing from WALTZDB. 
waltzdb.csv contains hexapeptides with binary annotation (1-amyloid or 0-non-amyloid). AggBERT_test and AggBERT_train essentially contain waltzdb entries, only filtered and split as supplied in the following paper: https://doi.org/10.1021/acs.jcim.3c00817 

Folder expression contains expression.csv in which PDB IDs are matched to UniProt identifiers along with the species to which a certain protein belongs to. The file also contains the expression value for human proteins. The expression value is based on consensus RNA expression data supplied by Human Protein Atlas and available online. Since our dataset contains non-human proteins as well, only part of the proteins are annotated with expression values.

Folder patches - contains individual csv files for each PDB entry, calculated by MolPatch Tool. MolPatch takes a PDB file and calculates the area of hydrophobic patch for each residue. Rank 0 means the largest hydrophobic patch. The value means that a residue is in a hydrophobic patch that is of that specific size.

folder benchmarking contains predictions for LHP, RHSA, and THSA from old shallow learning algorithms trained in the paper - How sticky are our proteins. In order to validate our model and compare to previous ones, we used the same test dataset and old predictions and plot the error threshold curves.

Folder miscellaneous contains files for data exploration and result interpretation/assessment. 
alphafold_molpatch.csv contains all human proteins predicted by alphafold and molpatch calculations based on alphafold structures.
consensus_mapped.txt is id mapping for expression data to be used
human_proteome.csv is predictions for all human proteins by patchprot
molpatch_ids_map.txt is id mapping for pdb to UniProt
molpatch_old.csv contains molpatch calculations for monomeric proteins with known X-ray structures
rna_tissue_consensus.tsv contains RNA expression values for each protein and tissues it is expressed in. Typically we analyse the largest values, essentially the expression potential for each protein. Notably, RNA values do not directly mean the abundance of a protein, but it is related and must be interpretted with caution.

summary.xlsx - contains the description of the extended data, all the features by which proteins are annotated. Some features are local (residue-based) some are global (one value per protein)