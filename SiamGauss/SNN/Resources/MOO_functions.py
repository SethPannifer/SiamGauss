import torch

def domination_check(objectives1, objectives2):
    output = [0,0,0]
    for index in range(0, len(objectives1)):
        if objectives1[index] > objectives2[index]:
            output[0] =1
        if objectives1[index] < objectives2[index]:
            output[2] =1
    if output == [1,0,1] or output ==[0,0,0]:
        return 0
    if output == [1,0,0]:
        return 1
    return -1

class mooDataset():
    def __init__(self, dataset_inputs, dataset_objective, sample_size = -1, quad_loss = True):
        self.dataset_inputs = dataset_inputs
        self.dataset_objective = dataset_objective
        self.quad_loss = quad_loss
        if sample_size != -1 :
            self.samples_created = sample_size
        else:
            self.samples_created = len(dataset_inputs)
        self.dominance_matrix = None
    
    def __len__(self):
        datapoints = self.samples_created
        return datapoints

    def __compute_dominancematrix_(self):
        n = len(self.dataset_objective)
        self.dominance_matrix = torch.zeros((n, n), dtype=torch.int8)

        # Vectorized comparison
        objs = self.dataset_objective
        for i in range(n):
            anchor_obj = objs[i].unsqueeze(0).expand(n, -1)  # shape: (n, num_objectives)
            greater = anchor_obj > objs
            less = anchor_obj < objs
            any_greater = greater.any(dim=1)
            any_less = less.any(dim=1)
            
            self.dominance_matrix[i] = torch.where(
                any_greater & (~any_less), 1,
                torch.where(any_less & (~any_greater), -1, 0)
            )

        return self.dominance_matrix
    
    def __getitem__(self, idx, print_instable = False):
        
        dataset_len = len(self.dataset_inputs)

        anchor_idx = idx % dataset_len
        anchor = self.dataset_inputs[anchor_idx]

        positive_examples = []
        neutral_examples = []
        negative_examples = []  
        for i in range(0,len(self.dataset_objective)):
            sample_check = domination_check(self.dataset_objective[anchor_idx],self.dataset_objective[i])
            
            if sample_check == 1:
                positive_examples.append(self.dataset_inputs[i])
            elif sample_check == 0:
                neutral_examples.append(self.dataset_inputs[i])
            else:
                negative_examples.append(self.dataset_inputs[i])
        if positive_examples == [] or neutral_examples == [] or negative_examples == []:
                if print_instable == True:
                    print('Dataset unstable, not enough samples provided')


        if self.quad_loss:
            return anchor, positive_examples, neutral_examples, negative_examples
        
   
    def __getitem_fast__(self, idx, print_instable = False):
        
        dataset_len = len(self.dataset_inputs)

        anchor_idx = idx % dataset_len
        anchor = self.dataset_inputs[anchor_idx]

        positive_examples = []
        neutral_examples = []
        negative_examples = []  
        for i in range(anchor_idx + 1,len(self.dataset_objective)):
            sample_check = domination_check(self.dataset_objective[anchor_idx],self.dataset_objective[i])
            
            if sample_check == 1:
                positive_examples.append(self.dataset_inputs[i])
            elif sample_check == 0:
                neutral_examples.append(self.dataset_inputs[i])
            else:
                negative_examples.append(self.dataset_inputs[i])
        if positive_examples == [] or neutral_examples == [] or negative_examples == []:
                if print_instable == True:
                    print('Dataset unstable, not enough samples provided')


        if self.quad_loss:
            return anchor, positive_examples, neutral_examples, negative_examples

    def __getitem_vectorised__(self, idx, print_instable=False):
        """
        Returns anchor and positive/neutral/negative examples using the full dataset.
        Ensures all anchors are compared against all other samples (except themselves).
        """
        dataset_len = len(self.dataset_inputs)
        anchor_idx = idx % dataset_len
        anchor = self.dataset_inputs[anchor_idx]

        # Compute dominance matrix if not already done
        if self.dominance_matrix is None:
            self.__compute_dominancematrix_()

        # Use all indices except the anchor itself
        mask_indices = torch.arange(dataset_len)
        mask_indices = mask_indices[mask_indices != anchor_idx]

        dominance_row = self.dominance_matrix[anchor_idx, mask_indices]
        inputs_subset = self.dataset_inputs[mask_indices]

        # Create masks
        positive_mask = dominance_row == 1
        negative_mask = dominance_row == -1
        neutral_mask = dominance_row == 0

        # Select examples
        positive_examples = inputs_subset[positive_mask]
        neutral_examples = inputs_subset[neutral_mask]
        negative_examples = inputs_subset[negative_mask]

        # Warn if any category is empty
        if len(positive_examples) == 0 or len(neutral_examples) == 0 or len(negative_examples) == 0:
            if print_instable:
                print(f'Dataset unstable for anchor {anchor_idx}, not enough samples provided')

        # Return anchor + example sets
        if self.quad_loss:
            return anchor, positive_examples, neutral_examples, negative_examples
