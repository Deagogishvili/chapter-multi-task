Some files are too large to be pushed to GitHub, therefore, all the files are provided via google drive: https://drive.google.com/drive/folders/17z34rgAw2nz4ywlF2G4CTJifcAI66lka?usp=sharing

Folder expression contains expression.csv in which PDB IDs are matched to UniProt identifiers along with the species to which a certain protein belongs to. The file also contains the expression value for human proteins. The expression value is based on consensus RNA expression data supplied by Human Protein Atlas and available online. Since our dataset contains non-human proteins as well, only part of the proteins are annotated with expression values.

Folder patches - contains individual csv files for each PDB entry, calculated by MolPatch Tool. MolPatch takes a PDB file and calculates the area of hydrophobic patch for each residue. Rank 0 means the largest hydrophobic patch. The value means that a residue is in a hydrophobic patch that is of that specific size.

folder benchmarking contains predictions for LHP from old shallow learning algorithms trained in the paper - How sticky are our proteins. In order to validate our model and compare to previous ones, we used the same test dataset and old predictions and plot the error threshold curves.

folder validation contains proteins from casp14 competition. Folder also contains AlphaFold predictions for these proteins, MolPatch calculations over the target PDB structures and AlphaFold predictions. We predicted LHP of these proteins with PatchProt and compared it with MolPatch calculations over AlphaFold structures.

summary.xlsx - contains the description of the extended data, all the features by which proteins are annotated. Some features are local (residue-based) some are global (one value per protein)
