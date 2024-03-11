import os
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt

def compute_eer_and_tdcf(cm_score_file):
    #asv_score_file = os.path.join(path_to_database, 'ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    # asv_data = np.genfromtxt(asv_score_file, dtype=str)
    # asv_sources = asv_data[:, 0]
    # asv_keys = asv_data[:, 1]
    # asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_sources = cm_data[:, 0]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 1].astype(np.float)

    other_cm_scores = -cm_scores

    # Extract target, nontarget, and spoof scores from the ASV scores
    # tar_asv = asv_scores[asv_keys == 'target']
    # non_asv = asv_scores[asv_keys == 'nontarget']
    # spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'])[0]

    print(cm_score_file)
    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(min(eer_cm, other_eer_cm) * 100))



    return min(eer_cm, other_eer_cm)


if __name__ == "__main__":
    cm_score_file = './FC1_result.txt'
    compute_eer_and_tdcf(cm_score_file)
    cm_score_file = './FC2_result.txt'
    compute_eer_and_tdcf(cm_score_file)
    cm_score_file = './FC3_result.txt'
    compute_eer_and_tdcf(cm_score_file)
    cm_score_file = './FC4_result.txt'
    compute_eer_and_tdcf(cm_score_file)
    cm_score_file = './FC5_result.txt'
    compute_eer_and_tdcf(cm_score_file)

