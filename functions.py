import random
import numpy as np
import torch





def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def stack_subject_windows(subject_id_list, epochs_data_per_subject, labels):
    """
    Returns:
    X: (N_windows_total, C, T)
    y: (N_windows_total,)
    Where each window inherits its subject's label.
    """
    X_list, y_list = [], []
    for sid in subject_id_list:
        X_sid = epochs_data_per_subject[sid]["Raw"]  # (E, C, T)
        y_sid = np.full((X_sid.shape[0],), labels[sid], dtype=np.float32)
        X_list.append(X_sid)
        y_list.append(y_sid)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y
