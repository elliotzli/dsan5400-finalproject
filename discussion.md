### Best SVM Parameters

| C   | Gamma | Kernel |
|-----|-------|--------|
| 10  | scale | rbf    |

### Best SVM Classification Report

| Rating | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| 1      | 0.78      | 0.77   | 0.77     |
| 2      | 0.67      | 0.24   | 0.36     |
| 3      | 0.66      | 0.44   | 0.53     |
| 4      | 0.52      | 0.58   | 0.55     |
| 5      | 0.76      | 0.85   | 0.80     |

### Best Random Forest Parameters

| max_depth | min_samples_split | n_estimators |
|-----------|-------------------|--------------|
| 30        | 5                 | 300          |

### Best Random Forest Classification Report

| Rating | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| 1      | 0.99      | 0.26   | 0.41     |
| 2      | 1.00      | 0.10   | 0.18     |
| 3      | 0.97      | 0.10   | 0.18     |
| 4      | 0.49      | 0.26   | 0.34     |
| 5      | 0.54      | 0.97   | 0.69     |

### Best KNN Parameters

| metric    | n_neighbors | weights  |
|-----------|-------------|----------|
| euclidean | 7           | distance |

### Best KNN Classification Report

| Rating | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| 1      | 0.75      | 0.44   | 0.55     |
| 2      | 0.63      | 0.21   | 0.32     |
| 3      | 0.73      | 0.30   | 0.43     |
| 4      | 0.56      | 0.36   | 0.44     |
| 5      | 0.58      | 0.90   | 0.71     |