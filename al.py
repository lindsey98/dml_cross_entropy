import os
os.chdir('/home/l/liny/ruofan/dml_cross_entropy/')

from src.model import get_model
from src.train import *
from src.data import *
from src.evaluate import *
from src.active_learning import *
import argparse

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

        model=training(model=model, 
                  labeldict=loaders.labeldict, 
                  loader=loaders.train, 
                  class_loss=class_loss, 
                  optimizer=optimizer,
                  scheduler=scheduler, 
                  epoch=epoch, 
                  logger=logger)

    # saving
    save_name = os.path.join(temp_dir, '{}_{}.pt'.format(cfg['MODEL']['arch'], cfg['DATA']['Name']))
    torch.save(model.state_dict(), save_name)
    return model
    
    
    
def al_loop(args) -> None:
    '''
        AL loop: Initial model --> AL selector --> Updated dataloader --> Retrain
                       |____________________________________________________|
    '''
    
    os.makedirs('log', exist_ok=True)

    ### initial model trained on initial training set ##### 
    with open(args.config_file) as file:
        initial_cfg = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() and not initial_cfg['Train']['cpu'] else 'cpu')
    al_strategy = initial_cfg['AL']['Strategy']
    exp_name = initial_cfg['DATA']['Name']
    seed = initial_cfg['Train']['seed']
    temp_dir = initial_cfg['Train']['temp_dir']
    model_arch = initial_cfg['MODEL']['arch']
    data_path = initial_cfg['DATA']['Path']['data_path']
    
    # logging
    with open('log/train_al_{}_{}.log'.format(exp_name, al_strategy), 'w'):
        pass
    logging.basicConfig(filename='log/train_al_{}_{}.log'.format(exp_name, al_strategy), level=logging.INFO)
    logger = logging.getLogger('trace')

    # load data
    loaders, recall_ks = get_loaders(initial_cfg)
    # load model
    initial_model = get_model(initial_cfg, loaders.num_classes)
    initial_model.load_state_dict(torch.load(os.path.join(temp_dir, '{}_{}.pt'.format(model_arch, exp_name))))
    initial_model.to(device)
    initial_model = nn.DataParallel(initial_model)
    
    cur_model = initial_model
    cur_cfg = initial_cfg.copy()
    logger.info(cur_cfg)
    
    # record dataset sizes 
    training_size = len(loaders.train_noshuffle.dataset)
    pool_size = len(loaders.pool.dataset)
    test_size = len(loaders.query.dataset)
    sampled_size = 0

    
    ##### main AL loop #####
    for it in range(1, args.al_iter+1):

        logger.info('Current AL iteration {}, Number of unlabelled samples been queried {}, Training size {} | Pool size {} | Test size {}, Number of distinct classes in gallery/training set {}'.format(it, sampled_size, training_size, pool_size, test_size, loaders.num_classes))
        
        # evaluate test matching acc!
        matching_acc, avg_tp, avg_fp, avg_tn, avg_fn = evaluator_aggregate(model=cur_model, loaders=loaders, recall_ks=recall_ks)
        print('Testing matching Acc at current iteration = {}, Avg TPs = {}, Avg FPs = {}, Avg TNs = {}, Avg FNs = {}'.format(matching_acc, avg_tp, avg_fp, avg_tn, avg_fn))
        logger.info('Testing matching Acc at current iteration = {}, Avg TPs = {}, Avg FPs = {}, Avg TNs = {}, Avg FNs = {}'.format(matching_acc, avg_tp, avg_fp, avg_tn, avg_fn))

        # limited budget
        if sampled_size >= training_size or sampled_size >= pool_size * 0.9:
            break

        # extract features
        all_pool_features, all_pool_labels, all_pool_samples = feature_extractor(model=cur_model, loaders=loaders.pool)
        all_gallery_features, all_gallery_labels, all_gallery_samples = feature_extractor(model=cur_model, loaders=loaders.train_noshuffle)
        
        # AL selector
        selected_indices, S = selector(all_gallery_features, all_gallery_labels, 
                                       all_pool_features, all_pool_labels, 
                                       strategy = al_strategy, 
                                       selected_k = int(0.1*training_size))

        selected_samples = np.asarray(all_pool_samples)[selected_indices]

        # update after AL
        # update original data file
        sampled_size += len(selected_samples.tolist())
        update_gallery = all_gallery_samples + selected_samples.tolist()
        update_pool = np.asarray(all_pool_samples)[~np.isin(np.arange(len(all_pool_samples)), selected_indices)]
        
        new_train_file = 'train_it{}_{}.txt'.format(it, al_strategy)
        new_pool_file = 'pool_it{}_{}.txt'.format(it, al_strategy)
        with open(os.path.join(data_path, new_train_file), 'w') as f:
            pass
        with open(os.path.join(data_path, new_train_file), 'a+') as f:
            for sample in update_gallery:
                f.write(sample[0].split(data_path)[1][1:] + ',' + str(sample[1]) + '\n')

        with open(os.path.join(data_path, new_pool_file), 'w') as f:
            pass
        with open(os.path.join(data_path, new_pool_file), 'a+') as f:
            for sample in update_pool:
                f.write(sample[0].split(data_path)[1][1:] + ',' + str(sample[1]) + '\n')

        # update cfg
        cur_cfg['DATA']['Path']['train_file'] = new_train_file
        cur_cfg['DATA']['Path']['pool_file'] = new_pool_file
        cur_cfg['DATA']['Name'] = 'cub_it{}_{}'.format(it, al_strategy)
        logger.info(cur_cfg)

        # update dataloader, update model
        torch.cuda.empty_cache() # empty cuda cache
        torch.manual_seed(seed)
        loaders, recall_ks = get_loaders(cur_cfg)
        training_size = len(loaders.train_noshuffle.dataset)
        pool_size = len(loaders.pool.dataset)
        test_size = len(loaders.query.dataset)
        
        torch.manual_seed(seed)
        cur_model = get_model(cur_cfg, loaders.num_classes)

        cur_model.to(device)
        cur_model = nn.DataParallel(cur_model)

        # retrain the model!
        cur_model = run(cfg=cur_cfg, loaders=loaders, recall_ks=recall_ks, model=cur_model, logger=logger)

        
def argparser() -> argparse.ArgumentParser:
    '''
        Parse arguments
    '''
    parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
    parser.add_argument("--config_file", required=True,
                      help="Path to config.yaml")
    parser.add_argument("--al_iter", default=20,
                      help="Number of AL iterations")
    return parser

    
if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    al_loop(args)
