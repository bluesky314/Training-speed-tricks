import numpy as np
import torch


def selective_backprop(losses,cutoff_perc=0.9):
    #https://github.com/oguiza/fastai_extensions/blob/master/03_BatchLossFilter.ipynb
    # for image-seg we get mask of indiv ce losses so take mean across dims:
    losses=torch.mean(losses,dim=(1,2))

    np_losses=np.array(losses.detach().cpu())
    cutoff =  cutoff_perc*np.sum(np_losses)
    sorted_losses=np.sort(np_losses)[::-1]
    idxs = np.argsort(np_losses)[::-1]
    cumilative_losses=sorted_losses.cumsum()
    
    chosen_idx=idxs[cumilative_losses<cutoff]
    number_chosen=np.sum(cumilative_losses<cutoff)

    model_loss=torch.mean(losses[chosen_idx])
    return(model_loss,number_chosen)
