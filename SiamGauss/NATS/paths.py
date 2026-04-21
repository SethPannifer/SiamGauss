from nats_bench import create

model_name = 'siamese_network.pt'
database_file = 'FILE_NAME_HERE.tar'
api = create(database_file, 'sss', fast_mode=True, verbose=False)
dataset = 'cifar10'