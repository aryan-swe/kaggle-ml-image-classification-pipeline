# Facial Feature Classification Challenge (ENGE707, Assessment 3)
Student 25299773. Eight-class cat and dog facial feature classification on Kaggle.

## Contents
- `25299773_Kaggle_Pipeline.ipynb`: the complete pipeline (Commits 1 to 14).
- `saved_results/`: checkpoints, training histories and saved predictions.
- `fast_resnet34_112.pt`, `reframe_resnet34_112.pt`: ResNet-34 weights (Commits 11b and 12).
- `submission_fast_tta7_plus_reframe.csv`: final Kaggle submission (leaderboard 0.9730).

## Reproduce
1. Create an environment and run `pip install -r requirements.txt`.
2. Download `train_part1.zip`, `train_part2.zip`, `test.zip`, `train_labels.csv` and `sample_submission.csv` from the Kaggle competition and put them next to the notebook.
3. Run all cells. Saved runs are loaded instead of retrained (`CONFIG["retrain"] = False`).

## Notes
- Pixel caches (tens of GB in total) are rebuilt on the first run and are not in the repository.
- Saved predictions are checked against checkpoint file times, so a fresh clone recomputes them (inference only, no training).
- The notebook imports `fcntl`, so it runs on macOS and Linux. ConvNeXt and ResNet ImageNet weights are downloaded by torchvision on first use.

## Commit history
Commits 1 to 5 were made in progression of the work, Commits 6 to 11c were made on one day and Commits 12 to 16 on a single day, after the notebook was finished. The CNN and ConvNeXt-Tiny checkpoints were trained in a rented NVIDIA B300 session at close to NZ$9 per hour, so I completed the notebook first and then committed each section in the order the work was done.