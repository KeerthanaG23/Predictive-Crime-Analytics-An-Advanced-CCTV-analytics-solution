from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils.utils import save_best_record

from tqdm import tqdm
from torch.multiprocessing import set_start_method
from tensorboardX import SummaryWriter
import option
args=option.parse_args()
from config import *
from models.mgfn import mgfn
from datasets.dataset import Dataset
from train import train
from test import test
import datetime





import datetime
# Function to sanitize paths
def sanitize_path(path):
    """
    Sanitize the given path by replacing invalid characters.
    """
    # Replace invalid characters with underscores
    invalid_chars = ['*', ':', '?', '"', '<', '>', '|', '\\', '/']
    for char in invalid_chars:
        path = path.replace(char, '_')
    return path

def save_config(save_path):
    # Sanitize the save_path to remove invalid characters
    sanitized_path = sanitize_path(save_path)

    # Ensure the path ends with a '/'
    path = sanitized_path + '/'
    
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Create a timestamped configuration file
    config_filename = "config_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(os.path.join(path, config_filename), 'w') as f:
        for key in vars(args).keys():
            f.write('{}: {}\n'.format(key, vars(args)[key]))

# Parse command-line arguments
args = option.parse_args()

# Construct the save path using sanitized arguments
savepath = './ckpt/{}_{}_{}_{}_{}_{}'.format(
    args.datasetname,
    args.feat_extractor,
    args.lr,
    args.batch_size,
    args.mag_ratio,
    args.comment
)

# Save the configuration
save_config(savepath)

# Initialize the SummaryWriter with the sanitized save path
log_writer = SummaryWriter(sanitize_path(savepath))

# Your training and testing code follows...


try:
     set_start_method('spawn')
except RuntimeError:
    pass


if __name__ == '__main__':
    args=option.parse_args()
    config = Config(args)
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)


    model = mgfn()
    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(model_ckpt)
        print("pretrained loaded")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.0005)
    test_info = {"epoch": [], "test_AUC": [], "test_PR":[]}

    best_AUC = -1
    best_PR = -1 # put your own path here

    for name, value in model.named_parameters():
        print(name)
    iterator = 0
    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):

        # for step in range(1, args.max_epoch + 1):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        cost, loss_smooth, loss_sparse = train(train_nloader, train_aloader, model, args.batch_size, optimizer,
                                                   device, iterator)
        log_writer.add_scalar('loss_contrastive', cost, step)

        if step % 1 == 0 and step > 0:
            auc, pr_auc = test(test_loader, model, args, device)
            log_writer.add_scalar('auc-roc', auc, step)
            log_writer.add_scalar('pr_auc', pr_auc, step)

            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_PR"].append(pr_auc)
            if args.datasetname == 'XD':
                if test_info["test_PR"][-1] > best_PR:
                    best_PR = test_info["test_PR"][-1]
                    torch.save(model.state_dict(), savepath + '/' + args.model_name + '{}-i3d.pkl'.format(step))
                    save_best_record(test_info, os.path.join(savepath + "/", '{}-step-AUC.txt'.format(step)))
            else:
                if test_info["test_AUC"][-1] > best_AUC :
                    best_AUC = test_info["test_AUC"][-1]
                    torch.save(model.state_dict(), savepath + '/' + args.model_name + '{}-i3d.pkl'.format(step))
                    save_best_record(test_info, os.path.join(savepath + "/", '{}-step-AUC.txt'.format(step)))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')