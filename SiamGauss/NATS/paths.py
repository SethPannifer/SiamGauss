from nats_bench import create

model_name = 'siamese_network.pt'
api = create('/Users/seth/Documents/PhD/Code/24.09 SNN/SNNforNAScode/SNN/Resources/NATS_DATASETS/NATS-tss-v1_0-3ffb9-simple', 'sss', fast_mode=True, verbose=False)
dataset = 'cifar10'