# A simplified version of the endlayers that isn't currently implemented
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#this code was from line 112 of energy_ood/CIFAR/test.py
to_np = lambda x: x.data.cpu().numpy()


def energyLossMod(loss,x,in_set,args):
    #This code was lines 192-196 of energy_ood/CIFAR/train.py and it is an addition to the training loss to account for energy.

    # cross-entropy from softmax distribution to uniform distribution
    if args.score == 'energy':
        Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
        Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())


    return loss


def energyScoreCalc(_score, output,args):
    #This code was from lines 133-134 of energy_ood/CIFAR/test.py
    if args.score == 'energy':
                    _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
    

    return _score
class argsc():
    def __init__(self):
        # EnergyOOD
        self.score = "energy"
        self.m_in = -1
        self.m_out = 0
        self.T = 1
        
args = argsc()

class EndLayers(nn.Module):
    def __init__(self, num_classes: int, cutoff=0.25, temperature=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.cutoff = float(cutoff)
        self.var_cutoff = 0.25
        self.temperature = temperature  # You can control the temperature for scaling the energy scores
        

    def energy_unknown(self, percentages: torch.Tensor, args):
        scores = []
        energyScoreCalc(scores, percentages, args)
        scores = torch.tensor(np.array(scores), device=percentages.device)
        # after converting it to a tensor, the wrong dimention is expanded
        scores = -scores.squeeze(dim=0).unsqueeze(dim=1)
        # This was to print the scores it was going to save
        # print(scores.sum()/len(scores))
        # Just store this for later
        # self.rocData[1] = -scores
        # self.Save_score.append(scores.mean())
        # once the dimentions are how we want them we test if it is above the cutoff
        scores = scores.less_equal(self.cutoff).to(torch.int)
        # Then we run precentages through a softmax to get a nice score
        percentages = torch.softmax(percentages, dim=1)
        # Finally we join the results as an unknown class in the output vector
        return torch.cat((percentages, scores), dim=1)
    
    def softMax_columns(self, percentages: torch.Tensor):
        batchsize = len(percentages)
        unknownColumn = torch.zeros(batchsize, device=percentages.device)
        return torch.cat((percentages, unknownColumn.unsqueeze(1)), dim=1)
    
    def var(self, logits):
        #Calculates variance
        return torch.var(torch.abs(logits), dim=1)
    
    def varMax(self, logits):
        return self.var(logits) < self.var_cutoff

    def forward(self, logits):
        # Step 1: Add energy-based unknowns
        logits_with_unknowns = self.energy_unknown(logits, args)
        
        # Step 2: Apply softmax, then add unknown column
        probs = F.softmax(logits, dim=1)
        softmax = self.softMax_columns(probs)        
        
        # Step 3: Get difference of top 2 classes
        top2 = torch.topk(probs, 2, dim=1).values
        diff = top2[:, 0] - top2[:, 1]  # Difference between top-2 probabilities
        thresh_mask = diff.less(0.5) #Stuff below threshold
        
        # Step 4: VarMax mask
        var_mask = self.varMax(logits)
        

        # Step 5: Apply the mask
        logits_with_unknowns[~(var_mask & thresh_mask)] = softmax[~(var_mask & thresh_mask)]

        return logits_with_unknowns
