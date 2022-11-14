"""
Predict
"""
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random, os, sys, torch, cv2, warnings
from glob import glob
from torch.utils.data import DataLoader

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

from modules.utils import load_yaml, save_yaml, get_logger
from modules.scalers import get_image_scaler
from modules.datasets import SegDataset
from models.utils import get_model
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    #! Load config
    config = load_yaml(os.path.join(prj_dir, 'config', 'predict_en.yaml'))
    train_config = load_yaml(os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'train.yaml'))
    train_config2 = load_yaml(os.path.join(prj_dir, 'results', 'train', config['train_serial2'], 'train.yaml'))
    train_config3 = load_yaml(os.path.join(prj_dir, 'results', 'train', config['train_serial3'], 'train.yaml'))

    #! Set predict serial
    pred_serial = config['train_serial'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create train result directory and set logger
    pred_result_dir = os.path.join(prj_dir, 'results', 'pred', pred_serial)
    pred_result_dir_mask = os.path.join(prj_dir, 'results', 'pred', pred_serial, 'mask')
    os.makedirs(pred_result_dir, exist_ok=True)
    os.makedirs(pred_result_dir_mask, exist_ok=True)

    # Set logger
    logging_level = 'debug' if config['verbose'] else 'info'
    logger = get_logger(name='train',
                        file_path=os.path.join(pred_result_dir, 'pred.log'),
                        level=logging_level)

    # Set data directory
    test_dirs = os.path.join(prj_dir, 'data', 'test')
    test_img_paths = glob(os.path.join(test_dirs, 'x', '*.png'))

    #! Load data & create dataset for train 
    test_dataset = SegDataset(paths=test_img_paths,
                            input_size=[train_config['input_width'], train_config['input_height']],
                            scaler=get_image_scaler(train_config['scaler']),
                            mode='test',
                            logger=logger)

    # Create data loader
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                shuffle=False,
                                drop_last=False)
    logger.info(f"Load test dataset: {len(test_dataset)}")

    # Load architecture
    model = get_model(model_str=train_config['architecture'])
    model = nn.DataParallel(model(
                classes=train_config['n_classes'],
                encoder_name=train_config['encoder'],
                encoder_weights=train_config['encoder_weight'],
                activation=train_config['activation'])).to(device)
    logger.info(f"Load model architecture: {train_config['architecture']}")

    model1 = get_model(model_str=train_config2['architecture'])
    model1 = nn.DataParallel(model1(
                classes=train_config2['n_classes'],
                encoder_name=train_config2['encoder'],
                encoder_weights=train_config2['encoder_weight'],
                activation=train_config2['activation'])).to(device)
    logger.info(f"Load model architecture: {train_config2['architecture']}")
    model2 = get_model(model_str=train_config2['architecture'])
    model2 = nn.DataParallel(model2(
                classes=train_config3['n_classes'],
                encoder_name=train_config3['encoder'],
                encoder_weights=train_config3['encoder_weight'],
                activation=train_config3['activation'])).to(device)
    logger.info(f"Load model architecture: {train_config3['architecture']}")

    #! Load weight
    check_point_path = os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'model.pt')
    check_point = torch.load(check_point_path)
    model.load_state_dict(check_point['model'])

    check_point_path1 = os.path.join(prj_dir, 'results', 'train', config['train_serial2'], 'model.pt')
    check_point1 = torch.load(check_point_path)
    model1.load_state_dict(check_point['model'])



    logger.info(f"Load model weight, {check_point_path}")

    # Save config
    save_yaml(os.path.join(pred_result_dir, 'train_config.yml'), train_config)
    save_yaml(os.path.join(pred_result_dir, 'predict_config.yml'), config)
    
    # Predict
    logger.info(f"START PREDICTION")

    model.eval()
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for batch_id, (x, orig_size, filename) in enumerate(tqdm(test_dataloader)):
            x = x.to(device, dtype=torch.float)
            y_pred1 = model(x)
            y_pred2 = model1(x)
            y_pred3 = model2(x)
            y_pred = (y_pred1 + y_pred2+y_pred3)/3
            y_pred_argmax = y_pred.argmax(1).cpu().numpy().astype(np.uint8)
            orig_size = [(orig_size[0].tolist()[i], orig_size[1].tolist()[i]) for i in range(len(orig_size[0]))]

            for filename_, orig_size_, y_pred_ in zip(filename, orig_size, y_pred_argmax):
                resized_img = cv2.resize(y_pred_, [orig_size_[1], orig_size_[0]], interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(pred_result_dir_mask, filename_), resized_img)
    logger.info(f"END PREDICTION")