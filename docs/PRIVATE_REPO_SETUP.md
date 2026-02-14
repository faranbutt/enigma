# Private Data Repository Setup

This document explains how to set up the private repository for storing ground truth labels.

## Overview

The evaluation system pulls validation and test labels from a **private GitHub repository** to prevent participants from accessing the ground truth.

## Setup Instructions

### 1. Create Private Repository

Create a new **private** repository named `enigma-private` with the following structure:

```
enigma-private/
├── test_labels.csv      # Ground truth for test set
├── valid_labels.csv     # Ground truth for validation set
└── README.md            # Documentation
```

### 2. Create Label Files

#### test_labels.csv
```csv
id,target
0,1
1,0
6,1
...
```

#### valid_labels.csv
```csv
id,target
241,1
244,0
254,1
...
```

**Note:** The IDs must match those in `data/test.csv` and `data/valid.csv` respectively.

### 3. Generate Labels from OGB

Use this Python script to generate the label files:

```python
from ogb.graphproppred import PygGraphPropPredDataset
import pandas as pd

# Load dataset
dataset = PygGraphPropPredDataset(name='ogbg-molbace')
split_idx = dataset.get_idx_split()

# Generate test labels
test_idx = split_idx['test'].tolist()
test_labels = [dataset[i].y.item() for i in test_idx]
test_df = pd.DataFrame({'id': test_idx, 'target': test_labels})
test_df.to_csv('test_labels.csv', index=False)

# Generate validation labels
valid_idx = split_idx['valid'].tolist()
valid_labels = [dataset[i].y.item() for i in valid_idx]
valid_df = pd.DataFrame({'id': valid_idx, 'target': valid_labels})
valid_df.to_csv('valid_labels.csv', index=False)

print(f"Test labels: {len(test_df)} entries")
print(f"Valid labels: {len(valid_df)} entries")
```

### 4. Create Personal Access Token (PAT)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name: `enigma-evaluation`
4. Set expiration (recommend: 90 days or no expiration for long-running competitions)
5. Select scopes:
   - `repo` (Full control of private repositories)
6. Click "Generate token"
7. **Copy the token immediately** (you won't see it again!)

### 5. Add Token to Main Repository Secrets

1. Go to the main `enigma` repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `PRIVATE_REPO_TOKEN`
4. Value: Paste the PAT you created
5. Click "Add secret"

### 6. Verify Setup

The GitHub Actions workflow will now:
1. Clone the private repository using the token
2. Copy the label files to the `data/` directory
3. Run the scoring script
4. Clean up (delete private data)

## Security Notes

- The PAT should have minimal permissions (only `repo` access)
- Consider rotating the token periodically
- The private repository should only be accessible to competition organizers
- Labels are never committed to the public repository
- GitHub Actions logs will not expose the token content

## Troubleshooting

### "Authentication failed"
- Check that the PAT is valid and not expired
- Verify the secret name matches `PRIVATE_REPO_TOKEN`
- Ensure the PAT has `repo` scope

### "Repository not found"
- Verify the private repository name is exactly `enigma-private`
- Check the repository owner matches the URL in the workflow
- Ensure the PAT has access to the repository

### "File not found"
- Verify `test_labels.csv` and `valid_labels.csv` exist in the private repo
- Check the file names are exactly correct (case-sensitive)
