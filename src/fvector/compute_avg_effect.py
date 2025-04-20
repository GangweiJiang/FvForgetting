import torch
import numpy as np
import os
import matplotlib.pyplot as plt

paths= "ag_news          ni1292_icl  ni1357_icl  ni339_icl  ni511_icl  object_v_concept_5 \
 alpaca                  choose_first_of_3        hellaswag   ni1310      ni1510      ni360      ni589      park-country \
 alphabetically_first_5  conll2003_location       mmlu_stem   ni1310_icl  ni1510_icl  ni360_icl  ni589_icl   \
      conll2003_person         ni1290      ni1343      ni195          ni618  \
 arc_c                              ni1290_icl  ni1343_icl  ni195_icl   ni363_icl  ni618_icl  \
 boolq                             ni1292          ni339            ob_count"
# paths = "conll2003_person  nq \
# coqa              ob_count \
# alphabetically_first_5  english-french    object_v_concept_5 \
# animal_v_object_5       park-country \
# antonym                  present-past \
# arc_c                   product-company \
# choose_first_of_3       ni1292_icl \
# conll2003_location      ni195_1024_icl"
paths=paths.split()

# paths = [
#     "lchat/hellaswag/sep",
#     "lchat/alpaca/sep",
#     # "lchat/arc_c/sep",
#     "lchat/park-country/sep",
#     "lchat/object_v_concept_5/sep",
#     # "lchat/ni618/sep",
#     # "lchat/ni1290/sep",
#     "lchat/ob_count/sep",
#     "lchat/coqa/sep"
# ]

name = "x"

indirect_paths = []
for i in paths:
    if "icl" not in i:
        i = f"lchat3/{i}/nsep"
        indirect_paths.append(f"/ossfs/workspace/gangwei/results/{i}_indirect_effect.pt")

indirect_effect = None
for indirect_effect_path in indirect_paths:
    print("Load Indirect Effects from", indirect_effect_path)
    if indirect_effect is None:
        indirect_effect = torch.load(indirect_effect_path)
    else:
        indirect_effect += torch.load(indirect_effect_path)
indirect_effect /= len(indirect_effect_path)

mean_indirect_effect = - indirect_effect[:,:,:,1].mean(dim=0)

n_top_heads=50
h_shape = mean_indirect_effect.shape 
print(h_shape)
topk_vals, topk_inds  = torch.topk(mean_indirect_effect.view(-1), k=n_top_heads, largest=True)
# top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(),4) for x in topk_vals]))
top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(),4) for x in topk_vals]))
top_heads = top_lh[:n_top_heads]

print(topk_vals[:n_top_heads],topk_inds[:n_top_heads])
for t in top_heads:
    print(t)
    # print("(",t[0],",", t[1],")")


def plot_similarity_heatmap(similarity, ax, title):
    cax = ax.matshow(similarity, cmap='RdYlBu', vmax=np.max(similarity), vmin=-np.max(similarity))
    # Add color bar
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    # Add title
    ax.set_title(title)
    # Remove x and y ticks
    ax.set_xticks(range(0,similarity.shape[1],4))
    ax.set_yticks(range(0,similarity.shape[0],4))

    ax.set_xlabel(f'head')
    ax.set_ylabel(f'layer')

result = mean_indirect_effect.cpu().numpy()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5))

plot_similarity_heatmap(result, axes, f'casual mediation')

fig.tight_layout()
# Show the plot
plt.savefig(os.path.join("/ossfs/workspace/gangwei/EIV/src/fvector/results", name+"_ie_heatmap.jpg"))
