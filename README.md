# Hydrophobic patch predictions by fine-tuning and multi-task learning of protein language models

Dea Gogishvili, Emmanuel Minois-Genin, Sanne Abeln

Protein property prediction from its primary sequence only is a valuable task to annotate proteins fast and with high accuracy. In addition to secondary structural components, solvent accessibility, disorder, and backbone geometry, there are various relevant protein properties with limited data availability. Multi-task deep learning offers a promising solution for addressing data gaps, simultaneously outperforming single-task methods. In this study, we harnessed existing deep learning architecture by NetSurfP3.0 and a recently released leading large language model ESM-2. Efficient fine-tuning of ESM-2 was achieved by leveraging recently developed parameter-efficient fine-tuning methods like LoRA and adapters. These approaches enabled comprehensive training of model layers without excessive parameters. Our study was expanded to address additional global tasks, including normalised protein expression, species, and large hydrophobic patch areas, the prediction of which was previously shown to be challenging. The results show that our fine-tuned ESM-2 model excelled at predicting global and local properties from protein sequences alone, setting a new standard for sequence-based protein property prediction. This research highlights the remarkable potential of large language models in predicting protein functional properties and the benefits of employing a multi-task learning strategy with partially annotated training datasets.

Folder baseline_models contain simple models trained on single tasks, such as aggregation or large hydrophobic patches
Folder data contains all the datafiles used and the link to the google drive folder for additional large files
Folder data_prep contains scripts for extending the dataset or subsetting for specific tasks + data/result exploration and analysis
Folder jobs contain bash scripts used to train models on a remote server on a specific GPU
Folder patchprot contains all the scripts necessary to train/test/utilise PatchProt

Large files are supplied via google drive: https://drive.google.com/drive/folders/1NcerEtJUn6eULDLdu2l-WPdzvTTw6mFE?usp=sharing