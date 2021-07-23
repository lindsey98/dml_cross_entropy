import os
os.chdir('/home/l/liny/ruofan/dml_cross_entropy/')

from src.model import get_model
from src.train import *
from src.data import *
from src.evaluate import *
from src.active_learning import *

def run(cfg: Dict, 
        loaders: MetricLoaders, 
        recall_ks: List[int], 
        model: nn.Module,
        logger) -> None:
    '''
        Main model training logic
    '''
    epochs = cfg['Train']['epochs']
    cpu = cfg['Train']['cpu']
    cudnn_flag = cfg['Train']['cudnn_flag']
    temp_dir = cfg['Train']['temp_dir']
    seed = cfg['Train']['seed']
    no_bias_decay = cfg['Train']['no_bias_decay']
    label_smoothing = cfg['Train']['label_smoothing']
    
    os.makedirs(temp_dir, exist_ok=True)

    class_loss = SmoothCrossEntropy(epsilon=label_smoothing) # get loss

    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(cfg=cfg, parameters=parameters, 
                                                   loader_length=len(loaders.train))

    torch.manual_seed(seed)
    for epoch in range(epochs):

        training(model=model, 
                  labeldict=loaders.labeldict, 
                  loader=loaders.train, 
                  class_loss=class_loss, 
                  optimizer=optimizer,
                  scheduler=scheduler, 
                  epoch=epoch, 
                  logger=logger)

    # saving
    save_name = os.path.join(temp_dir, '{}_{}_{}.pt'.format(cfg['MODEL']['arch'], cfg['DATA']['Name'], cfg['AL']['Strategy']))
    torch.save(model.state_dict(), save_name)
    
    
if __name__ == '__main__':
   

    ### initial model trained on initial training set ##### 
    with open('./configs/cub200.yaml') as file:
        initial_cfg = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() and not initial_cfg['Train']['cpu'] else 'cpu')
#     device = 'cpu'

    os.makedirs('log', exist_ok=True)
    with open('log/train_al_{}_{}.log'.format(initial_cfg['DATA']['Name'], initial_cfg['AL']['Strategy']), 'w'):
        pass
    logging.basicConfig(filename='log/train_al_{}_{}.log'.format(initial_cfg['DATA']['Name'], initial_cfg['AL']['Strategy']), level=logging.INFO)
    logger = logging.getLogger('trace')

    seed = initial_cfg['Train']['seed']
    loaders, recall_ks = get_loaders(initial_cfg)
    initial_model = get_model(initial_cfg, loaders.num_classes)
    initial_model.load_state_dict(torch.load(os.path.join(initial_cfg['Train']['temp_dir'], 
                                                          '{}_{}.pt'.format(initial_cfg['MODEL']['arch'], initial_cfg['DATA']['Name']))))
    initial_model.to(device)
    initial_model = nn.DataParallel(initial_model)

    initial_training_size = len(loaders.train_noshuffle.dataset)
    initial_pool_size = len(loaders.pool.dataset)
    sampled_size = 0
    cur_model = initial_model
    cur_cfg = initial_cfg.copy()
    logger.info(cur_cfg)
    
    
    ##### main AL loop #####
    for it in range(20):

        logger.info('Current AL iteration {}, Number of unlabelled samples been queried {}, Number of distinct classes in gallery/training set {} '.format(it, sampled_size, loaders.num_classes))

        # limited budget
        if sampled_size >= initial_training_size or sampled_size >= initial_pool_size * 0.9:
            break

        # extract features
        all_pool_features, all_pool_labels, all_pool_samples = feature_extractor(model=cur_model, loaders=loaders.pool)
        all_gallery_features, all_gallery_labels, all_gallery_samples = feature_extractor(model=cur_model, loaders=loaders.train_noshuffle)

        # evaluate !
        matching_acc = evaluator_aggregate(model=cur_model, loaders=loaders, recall_ks=recall_ks)
        print('Testing matching Acc at current iteration = {}'.format(matching_acc))
        logger.info('Testing matching Acc at current iteration = {}'.format(matching_acc))

        # AL selector
        selected_indices, S = selector(all_gallery_features, all_gallery_labels, 
                                       all_pool_features, all_pool_labels, 
                                       strategy=cur_cfg['AL']['Strategy'], 
                                       selected_k=int(0.1*initial_training_size))

        selected_samples = np.asarray(all_pool_samples)[selected_indices]

        # update after AL
        # update original data file
        sampled_size += len(selected_samples.tolist())
        update_gallery = all_gallery_samples + selected_samples.tolist()
        update_pool = np.asarray(all_pool_samples)[~np.isin(np.arange(len(all_pool_samples)), selected_indices)]

        with open(os.path.join(cur_cfg['DATA']['Path']['data_path'], 'train_it{}.txt'.format(it)), 'w') as f:
            pass
        with open(os.path.join(cur_cfg['DATA']['Path']['data_path'], 'train_it{}.txt'.format(it)), 'a+') as f:
            for sample in update_gallery:
                f.write(sample[0].split(cur_cfg['DATA']['Path']['data_path'])[1][1:] + ',' + str(sample[1]) + '\n')

        with open(os.path.join(cur_cfg['DATA']['Path']['data_path'], 'pool_it{}.txt'.format(it)), 'w') as f:
            pass
        with open(os.path.join(cur_cfg['DATA']['Path']['data_path'], 'pool_it{}.txt'.format(it)), 'a+') as f:
            for sample in update_pool:
                f.write(sample[0].split(cur_cfg['DATA']['Path']['data_path'])[1][1:] + ',' + str(sample[1]) + '\n')

        # update cfg
        cur_cfg['DATA']['Path']['train_file'] = 'train_it{}.txt'.format(it)
        cur_cfg['DATA']['Path']['pool_file'] = 'pool_it{}.txt'.format(it)
        cur_cfg['DATA']['Name'] = 'cub_it{}'.format(it)
        logger.info(cur_cfg)

        # update dataloader, update model
        torch.manual_seed(seed)
        loaders, recall_ks = get_loaders(cur_cfg)
        print(loaders.num_classes)
        torch.manual_seed(seed)
        cur_model = get_model(cur_cfg, loaders.num_classes)

        cur_model.to(device)
        cur_model = nn.DataParallel(cur_model)

        # retrain!
        run(cfg=cur_cfg, loaders=loaders, recall_ks=recall_ks, model=cur_model, logger=logger)




