# EMC-DWES

[Automatic frequency-based feature selection using discrete weighted evolution strategy](https://www.sciencedirect.com/science/article/pii/S1568494622007487#!)


EMC-DWES is a hybrid feature selection method. It combines the filter feature selection (EMC) with the wrapper feature selection (DWES). EMC is the improved version of MC (You can read this article for MC: [Frequency based feature selection method using whale algorithm](https://www.sciencedirect.com/science/article/pii/S0888754318304245)). EMC-DWES.py also contains related codes for MC (In case you want to run MC and compare the results with EMC). You can also download the 9 datasets. Then, you need to specify the valid string path based on the location of your datasets in your storage. For your convenience, the results of EMC on individual datasets are uploaded as well. You can simply use those files (download alphaDataset.npy in alphas folder and rename it to alpha.npy before importing in pyhton) and ignore running EMC section (lines 132-170). Note that for bigger datasets calculating EMC can be time consuming.


TYPO: According to what has mentioned in the paper all the features in the dataset are sorted according to their calculated EMC measure. Then, 95% of the worst features are discarded and only the best remaining 5% are passed to the DWES phase. This is clearly reflected in the text and therefore Eq. 14 in the paper is a typo and should be changed from m′ = 0.95 × m     to   m′ = 0.05 × m  to correctly reflect the information in the text.


