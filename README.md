# Hydrophobic patch predictions by multi-task learning and fine-tuning protein foundation models

Dea Gogishvili, Emmanuel Minois-Genin, Jan van Eck, Sanne Abeln

### Abstract

Hydrophobic patches on protein surfaces play important functional roles in protein-protein and protein-ligand interactions. Large hydrophobic surfaces are also involved in the progression of aggregation diseases. Predicting exposed hydrophobic patches from a protein sequence has been shown to be a difficult task. Multi-task deep learning offers a promising solution for addressing data gaps, simultaneously outperforming single-task methods. In this study, we harnessed existing deep learning architecture by NetSurfP-3.0 and a recently released leading large language model ESM-2. Efficient fine-tuning of ESM-2 was achieved by leveraging a recently developed parameter-efficient fine-tuning method. This approach enabled comprehensive training of model layers without excessive parameters and without the need to include a computationally expensive multiple sequence analysis. We explored several related tasks, at local (residue) and global (protein) levels, to improve the representation of the model. As a result, our fine-tuned ESM-2 model, PatchProt, cannot only predict hydrophobic patch areas but also outperforms existing methods at predicting primary tasks, including secondary structure and surface accessibility predictions. Importantly, our analysis shows that including related local tasks can improve predictions on more difficult global tasks. This research sets a new standard for sequence-based protein property prediction and highlights the remarkable potential of fine-tuning foundation models enriching the model representation by training over related tasks.

Keywords: Multi-task learning, Sequence-based protein property prediction, Protein language model, ESM, LoRA.

![image](/figures/model.png)
Figure 1. Model architecture. The model takes protein sequence as input and predicts both global and local protein properties. The model consists of an embedding output from ESM-2 protein language model and the downstream architecture similar to NetSurfP-3.0. Additionally, a parameter-efficient fine-tuning strategy was implemented. The decoding head consists of a residual block with two convolutional neural network (CNN) layers and a two-layer bidirectional long short-term memory (BiLSTM) network. The output is fed into a fully connected layer to provide predictions for all residues- and protein-level tasks.

Details:

- Folder data contains all the datafiles used and the link to the google drive folder for additional large files
- Folder data_prep contains scripts for extending the dataset
- Folder jobs contain bash scripts used to train models on a remote server on a specific GPU
- Folder patchprot contains all the scripts necessary to train/test/utilise PatchProt

Large files are supplied via google drive: https://drive.google.com/drive/folders/1NcerEtJUn6eULDLdu2l-WPdzvTTw6mFE?usp=sharing

Folder "data" contains all the files used for training and testing our models. 
- Source datasets from NetSurfP
- Folder "extended" contains the same datasets annotated with LHP values. The same folder also contains datasets with only proteins with LHP annotations to be able to train models for only LHP predictions. 