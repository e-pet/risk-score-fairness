import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from metrics import avg_underrepr

# The following example illustrates how a model can have equal AUC across groups while being representation-unfair
# at different thresholds.
#
# We have two groups, A and B. A has base rate 2/3, B has base rate 1/3. Both are equally large.
#
# We will construct two beta distributions for the true risks and assume a model that perfectly predicts those.
#
# These are constructed such that
# 1) AUROC will be equal,
# 2) they are both better than chance everywhere,
# 3) they have different score distributions, leading to different group representation in the selected set at different
#    thresholds, and
# 4) the classifiers can be well-calibrated. (The latter requires that the ROC curves are convex.)
#
# Construct two beta distributions, one with incidence E[R] = 1/3 and one with incidence E[R]=2/3.
# Both are two have AUROC = 0.75, corresponding to a moderately good model.
#
# E[X], with X beta-distributed, is given by alpha/(alpha+beta), both > 0.
# So we can reparameterize this as beta = (1-p)/p * alpha, where p=E[R].

N_for_alpha_est = 1000000

p_A = 2/3
p_B = 1/3


def get_auroc_from_alpha(alpha, p):
    scores = np.random.beta(alpha, (1-p)/p*alpha, size=(N_for_alpha_est,))
    y = np.random.binomial(n=1, p=scores)
    return roc_auc_score(y, scores)


#res_A = minimize_scalar(lambda alpha: (0.75 - get_auroc_from_alpha(alpha, p=p_A))**2, bounds=[1e-2, 20],
#                        method="bounded", tol=1e-7)
#res_B = minimize_scalar(lambda alpha: (0.75 - get_auroc_from_alpha(alpha, p=p_B))**2, bounds=[1e-2, 20],
#                        method="bounded", tol=1e-7)
#alpha_A = res_A.x
#alpha_B = res_B.x

alpha_A = 3.1956183608423134
alpha_B = 1.6046978496458115

print(f'AUROC {get_auroc_from_alpha(alpha_A, p_A)} @ {alpha_A}')
print(f'AUROC {get_auroc_from_alpha(alpha_B, p_B)} @ {alpha_B}')

N_for_plots = 100000

scores_A = np.random.beta(alpha_A, (1-p_A)/p_A*alpha_A, size=(N_for_plots,))
y_A = np.random.binomial(n=1, p=scores_A)
scores_B = np.random.beta(alpha_B, (1-p_B)/p_B*alpha_B, size=(N_for_plots,))
y_B = np.random.binomial(n=1, p=scores_B)
P_A = sum(y_A)
P_B = sum(y_B)

# --- Plot things for the paper

fontsize = 8
params = {
    'axes.labelsize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}'
}
plt.rcParams.update(params)
cmap = sns.color_palette(cc.glasbey_category10, as_cmap=True)

fig, axes = plt.subplots(1, 3, dpi=600)
# textwidth in inches: 5.95114in textheight in inches: 7.75024in -- but we also need space for the caption
fig.set_size_inches(5.95114, 1.8)


scores_all = np.concatenate([scores_A, scores_B])
group_A_msk = np.concatenate([np.ones_like(scores_A), np.zeros_like(scores_B)])
group_B_msk = np.concatenate([np.zeros_like(scores_A), np.ones_like(scores_B)])
y_all = np.concatenate([y_A, y_B])

data_df = pd.DataFrame({"y_pred_proba": scores_all, "group": group_B_msk, "y_true": y_all})
h = sns.kdeplot(data=data_df, x="y_pred_proba", hue="group", hue_order=[0, 1], ax=axes[0],
                common_norm=False, clip=[0, 1], cut=0, palette=cmap, legend=False)
plt.sca(axes[0])
plt.xlim([0, 1])
plt.xlabel(r"True risk $\rho=R$")
plt.ylabel(r"Density $p(\rho \mid G)$")
axes[0].yaxis.grid(True, linestyle='--')
plt.title('(True) risk score distribution')

fpr_A, tpr_A, _ = roc_curve(y_A, scores_A)
axes[1].plot(fpr_A, tpr_A, label="Group A")
fpr_B, tpr_B, _ = roc_curve(y_B, scores_B)
axes[1].plot(fpr_B, tpr_B, label="Group B")
plt.sca(axes[1])
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("FPR")  # = 1 - Specificity
plt.ylabel("TPR")  # = Sensitivity = Recall
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('')
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.grid(visible=True)
plt.title('ROC diagram')

avg_under_repr_A, threshs_A, rel_repr_A = avg_underrepr(scores_all, group_A_msk == 1, risk_quantile_bounds=None,
                                                        target_repr=P_A/(P_A+P_B), return_datapoints=True)
avg_under_repr_B, threshs_B, rel_repr_B = avg_underrepr(scores_all, group_B_msk == 1, risk_quantile_bounds=None,
                                                        target_repr=P_B/(P_A+P_B), return_datapoints=True)

axes[2].plot(threshs_A, rel_repr_A, color=cmap[0], label=f"Group A, EUR={avg_under_repr_A:.2f}")
axes[2].plot(threshs_B, rel_repr_B, color=cmap[1], label=f"Group B, EUR={avg_under_repr_B:.2f}")

axes[2].plot([0, 1], [1, 1], 'k--', label="ideal")
axes[2].set(xlabel=r"$\tau$", ylabel=r"$P(G \mid \hat{Y}{=}1) / P(G \mid Y{=}1)$")
axes[2].grid(visible=True)
plt.sca(axes[2])
plt.title('Relative representation in selected set')
plt.tight_layout()

fig.savefig(f"figs/auc_ranking_example.pdf", bbox_inches="tight", pad_inches=0)

print(avg_under_repr_A)
print(avg_under_repr_B)

plt.figure()
sns.displot(data=data_df, x="y_pred_proba", hue="group", kind="kde", col="y_true", hue_order=[0, 1],
            common_norm=False, clip=[0, 1], cut=0, palette=cmap, legend=False)

plt.show()
