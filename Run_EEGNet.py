import mne
from pathlib import Path
import os
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
mne.set_log_level("ERROR")
import re
SAVE_DIR = Path("Multi_run_models")
SAVE_DIR.mkdir(exist_ok=True)
import random


from EEGNet import EEGNetBinary, EEGWindowDataset
from functions import set_seed, stack_subject_windows


rng = random.Random(42)






# -------------------------------------------------
# Parameters, Declarations and Paths
# -------------------------------------------------

# EEGNet hyperparameters
Window         = 6
overlap_frac   = 0
overlap_sec    = Window * overlap_frac
epoch_duration = Window

learning_rate  = 5.55e-05
weight_decay   = 7.7e-6
dropout        = 0.35

kernel_length  = 32
F1             = 16
D              = 3

batch_size     = 16
patience       = 10

num_epochs     = 25   
val_split = 0.2
l2_reg = 1e-6


# Lists and dicts used 

raw_data_per_subject = {}
labels = {}

adhd_subjects = []
control_subjects = []

bal_accs_this_trial = []
epochs_data_per_subject = {}


# Folder-specific paths and filenames. In my directory, the data is example: New_data/out_fif/v196_class-ADHD_raw.fif

DATA_DIR = Path("New_data/out_fif")  
TASK_KEY = "Raw"     

pattern = re.compile(
    r"v(?P<vid>\d+)(?:p)?_class-(?P<class>ADHD|Control)_raw\.fif$"
)

save_model = False                  # Descides whether the trained model should be saved locally for later examination
output_file = "Multirun.txt"        # Textsummary of each finished LOSO-run







# -------------------------------------------------
# Data loading from file
# -------------------------------------------------
for filepath in sorted(DATA_DIR.glob("*.fif")):
    match = pattern.match(filepath.name)
    if not match:
        continue

    vid = int(match.group("vid"))
    group = match.group("class")

    subject = f"sub-P{vid:03d}"



    raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    raw = raw.copy().crop(tmin=5)                                           # Crop 5 seconds
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)


    # -------------------------------------------------
    # Store data
    # -------------------------------------------------
    raw_data_per_subject[subject] = {TASK_KEY: raw}

    if group == "ADHD":
        labels[subject] = 1
        adhd_subjects.append(subject)
    else:
        labels[subject] = 0
        control_subjects.append(subject)

adhd_subjects.sort()
control_subjects.sort()



subject_float_predictions = {
    sid: {"subject": sid, "ADHD": labels[sid]}
    for sid in sorted(labels.keys())
}






# -------------------------------------------------
# Definitions of runs [Min bandpass limit, max bandpass limit, name] and the number of randomize seeds
# -------------------------------------------------

# Delta = 0-4 Hz
# Theta = 4-8 Hz
# Alpha = 8-12 Hz
# Beta = 12-30 Hz
# Gamma > 30 Hz

runs = [[0.5,40,"Full_Bandpass_0.5-40Hz"],]
""""     
        [0.5,8,"Delta_Theta_0.5-8Hz"],
        
        [8,12,"Alpha_8-12Hz"],
        [8,20,"Alpha+low_beta8-20Hz"],
        
        [12,30,"Beta_12-30Hz"],
        [12,40,"Gamma+beta12-40Hz"],

        [20,40,"Gamma+High_beta20-40Hz"],
        [30,40,"Gamma30-40Hz"],]

"""

seeds = [41,42,43]
seeds = [1]












# -------------------------------------------------
# Main run 
# -------------------------------------------------

def main():

    for run in runs:
        for seed in seeds:

            set_seed(seed)
            minHz = run[0]
            maxHz = run[1]
            Name = run[2]
            
            with open(output_file, "a") as f:
                f.write(f"Run {Name} with seed {seed}")


            # ------------------------------------------------------------
            # Preprocessing 
            # ------------------------------------------------------------
                        
            epochs_data_per_subject = {}

            for subj_id, task_dict in raw_data_per_subject.items():
                subj_epochs = {}

                # Bandpass filter
                raw = task_dict["Raw"].copy()
                picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
                raw.filter(l_freq=minHz, h_freq=maxHz, picks=picks, verbose=False) 


                # Split data into windows
                events = mne.make_fixed_length_events(
                    raw,
                    duration=Window,
                    overlap=overlap_sec
                )                        
                epochs = mne.Epochs(raw, events=events, tmin=0.0, tmax=epoch_duration,
                                    baseline=None, preload=True, verbose=False)

                data_array = epochs.get_data() 


                # Normalization of each window
                mean = data_array.mean(axis=2, keepdims=True)
                std = data_array.std(axis=2, keepdims=True)
                data_array = (data_array - mean) / (std + 1e-6)


                # Store all preprocessed data
                subj_epochs["Raw"] = data_array
                epochs_data_per_subject[subj_id] = subj_epochs


                    
            # -----------------------------
            # 0) Collect per-subject arrays
            # -----------------------------

            subjects = sorted(list(epochs_data_per_subject.keys()))
            n_subjects = len(subjects)


            # grab dimensions from first subject
            sample_X = epochs_data_per_subject[subjects[0]]["Raw"]      #(n_windows, n_channels, n_samples)
            n_channels = sample_X.shape[1]
            window_size_samples = sample_X.shape[2]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            # --------------------------------------------
            # 1) LOSO loop bookkeeping. sid = subject_id
            # --------------------------------------------
            subject_probabilities = {}   # sid -> probs per window
            subject_true_labels   = {}   # sid -> true label
            window_accs           = {}   # sid -> window-level accuracy vs subject label


            # --------------------------------------------
            # 2) LOSO folds
            # --------------------------------------------

            for _, test_sid in enumerate(subjects):
                print(test_sid)


                # ------------------------------------------------------------
                # 2.1) Split subjects: 1 test, rest train+val
                # ------------------------------------------------------------
                trainval_sids = [sid for sid in subjects if sid != test_sid]


                # ------------------------------------------------------------
                # 2.2) Inner train/val split (SUBJECT-WISE)
                # ------------------------------------------------------------
                trainval_labels = np.array([labels[sid] for sid in trainval_sids], dtype=int)

                train_sids, val_sids = train_test_split(
                        trainval_sids,
                        test_size=0.2,
                        random_state=42,
                        shuffle=True,
                        stratify=trainval_labels)


                # ------------------------------------------------------------
                # 2.3) Build window datasets by stacking subject epochs
                # ------------------------------------------------------------
                X_train, y_train = stack_subject_windows(train_sids, epochs_data_per_subject, labels)
                X_val,   y_val   = stack_subject_windows(val_sids, epochs_data_per_subject, labels)
                X_test,  y_test  = stack_subject_windows([test_sid], epochs_data_per_subject, labels)

                train_loader = DataLoader(EEGWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
                val_loader   = DataLoader(EEGWindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
                test_loader  = DataLoader(EEGWindowDataset(X_test, y_test), batch_size=batch_size, shuffle=False)


                # ------------------------------------------------------------
                # 2.4) New EEGNet for this fold
                # ------------------------------------------------------------
                model = EEGNetBinary(
                    n_channels=n_channels,
                    window_size=window_size_samples,
                    F1=F1,
                    D=D,
                    kernel_length=kernel_length,
                    dropout_p=dropout            
                    ).to(device)


                # ------------------------------------------------------------
                # 2.5) Loss with class imbalance handling
                # ------------------------------------------------------------
                ADHD = (y_train == 1).sum()
                Control = (y_train == 0).sum()
                
                pos_weight_val = Control / max(ADHD, 1)

                pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


                # ------------------------------------------------------------
                # 2.6) Training loop with early stopping on validation loss
                # ------------------------------------------------------------
                best_val_loss = np.inf
                epochs_no_improve = 0
                best_state = None

                for epoch in range(num_epochs):
                    model.train()

                    train_losses = []
                    train_probs, train_targets = [], []

                    for Xb, yb in train_loader:
                        Xb = Xb.to(device)
                        yb = yb.to(device).unsqueeze(1) 

                        optimizer.zero_grad()
                        logits = model(Xb)             
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()

                        train_losses.append(loss.item())
                        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                        train_probs.extend(probs)
                        train_targets.extend(yb.detach().cpu().numpy().flatten())



                    # Validate and early stopping
                    model.eval()
                    val_losses = []
                    val_probs, val_targets = [], []

                    with torch.no_grad():
                        for Xb, yb in val_loader:
                            Xb = Xb.to(device)
                            yb = yb.to(device).unsqueeze(1)

                            logits = model(Xb)
                            loss = criterion(logits, yb)
                            val_losses.append(loss.item())

                            probs = torch.sigmoid(logits).cpu().numpy().flatten()
                            val_probs.extend(probs)
                            val_targets.extend(yb.cpu().numpy().flatten())

                    val_loss = float(np.mean(val_losses))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        break

                # Restore best fold weights
                model.load_state_dict(best_state)




                # Save trained model for later inspection
                if save_model:
                    save_path = SAVE_DIR / f"{Name}_seed{seed}_EEGNet_test-{test_sid}.pt"

                    torch.save({
                        "test_subject": test_sid,
                        "window_length_sec": Window,
                        "model_class": "EEGNetBinary",
                        "model_kwargs": {
                            "n_channels": n_channels,
                            "window_size": window_size_samples,
                            "F1": F1,
                            "D": D,
                            "kernel_length": kernel_length,
                            "dropout_p": dropout,
                        },
                        "state_dict": model.state_dict(),
                        "val_sids": val_sids,
                        "pos_weight": float(pos_weight_val),
                    }, save_path)


                # ------------------------------------------------------------
                # 2.7) Test: window-level probabilities for held-out subject
                # ------------------------------------------------------------
                model.eval()
                all_probs = []
                with torch.no_grad():
                    for Xb, yb in test_loader:
                        Xb = Xb.to(device)
                        logits = model(Xb)
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        all_probs.extend(probs)

                all_probs = np.array(all_probs)
                true_label = int(labels[test_sid])



                subject_probabilities[test_sid] = all_probs
                subject_true_labels[test_sid] = true_label

                window_preds = (all_probs >= 0.5).astype(int)
                
                window_acc = float(np.mean(window_preds == true_label))
                window_accs[test_sid] = window_acc

                #print(f"  Test subject {test_sid}: {len(all_probs)} windows | "
                #    f"true={true_label} | mean prob={all_probs.mean():.3f} | "
                #    f"window-acc={window_acc:.3f}")





            # ------------------------------------------------------------
            # 3) Subject-level decision. All predictions are stored in a CSV-file
            # ------------------------------------------------------------

            y_true_subject = []
            y_pred_subject = []

            for sid in subjects:
                probs = subject_probabilities[sid]
                mean_prob = probs.mean()
                col_name = f"{Name}_seed{seed}"
                subject_float_predictions[sid][col_name] = round(mean_prob,3)
  
                pred_label = 1 if mean_prob >= 0.5 else 0

                y_true_subject.append(subject_true_labels[sid])
                y_pred_subject.append(pred_label)


                csv_path = f"{Name}.csv"
                rows = []
                true = subject_true_labels[sid] 

                total_windows = len(probs)

                for window_number, prob in enumerate(probs):
                    rows.append({
                        "Run": Name, 
                        "Seed": seed,
                        "subject": sid, 
                        "window_number": window_number, 
                        "total_windows": total_windows,
                        "prob": float(prob),
                        "mean_prob": mean_prob,
                        "ADHD": true,
                        })
                df_new = pd.DataFrame(rows)


                df_new.to_csv(
                csv_path,
                mode="a",
                header=not os.path.exists(csv_path),
                index=False
                )


            #----------------------------------------------
            # 4) Overall statistics from the run
            # ------------------------------------------------------------

            macro_window_accuracy = float(np.mean([window_accs[sid] for sid in subjects]))


            cm = confusion_matrix(y_true_subject, y_pred_subject, labels=[0, 1])
            TN, FP, FN, TP = cm.ravel()


            tpr = TP/(TP+FN) if (TP+FN)>0 else 0.0
            tnr = TN/(TN+FP) if (TN+FP)>0 else 0.0
            balanced_accuracy = 0.5*(tpr+tnr)

            bal_accs_this_trial.append(balanced_accuracy)



            with open(output_file, "a") as f:
                f.write("\n")
                f.write(f"{Name}\n")
                f.write(f"Macro window accuracy (each subject equally): {macro_window_accuracy:.3f}\n")
                f.write(f"Balanced accuracy: {balanced_accuracy:.3f}\n")
                f.write("Confusion matrix:\n")
                f.write(f"{confusion_matrix(y_true_subject, y_pred_subject)}\n")


if __name__ == "__main__":
    main()
