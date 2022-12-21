# -- Built-in modules -- #
import os

# -- Third-part modules -- #
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm.notebook import tqdm

# --Proprietary modules -- #
from functions import chart_cbar, r2_metric, f1_metric, compute_metrics
from loaders import AI4ArcticChallengeTestDataset, AI4ArcticChallengeDataset
from unet import UNet
from utils import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES, colour_str



os.environ['AI4ARCTIC_DATA'] = 'C:\\Users\\Ultimate Gaming Comp\\Documents\\Ice_Data'  # Fill in directory for data location.
os.environ['AI4ARCTIC_ENV'] = 'C:\\Users\\Ultimate Gaming Comp\\Documents\\Ice_Challenge\AI4ArcticSeaIceChallenge'  # Fill in directory for environment with Ai4Arctic get-started package.

train_options = {
    # -- Training options -- #
    'path_to_processed_data': os.environ['AI4ARCTIC_DATA'],  # Replace with data directory path.
    'path_to_env': os.environ['AI4ARCTIC_ENV'],  # Replace with environmment directory path.
    'lr': 0.0001,  # Optimizer learning rate.
    'epochs': 50,  # Number of epochs before training stop.
    'epoch_len': 500,  # Number of batches for each epoch.
    'patch_size': 256,  # Size of patches sampled. Used for both Width and Height.
    'batch_size': 8,  # Number of patches for each batch.
    'loader_upsampling': 'nearest',  # How to upscale low resolution variables to high resolution.

    # -- Data prepraration lookups and metrics.
    'train_variables': SCENE_VARIABLES,  # Contains the relevant variables in the scenes.
    'charts': CHARTS,  # Charts to train on.
    'n_classes': {  # number of total classes in the reference charts, including the mask.
        'SIC': SIC_LOOKUP['n_classes'],
        'SOD': SOD_LOOKUP['n_classes'],
        'FLOE': FLOE_LOOKUP['n_classes']
    },
    'pixel_spacing': 80,  # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
    'train_fill_value': 0,  # Mask value for SAR training data.
    'class_fill_values': {  # Mask value for class/reference data.
        'SIC': SIC_LOOKUP['mask'],
        'SOD': SOD_LOOKUP['mask'],
        'FLOE': FLOE_LOOKUP['mask'],
    },

    # -- Validation options -- #
    'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
        'SIC': {
            'func': r2_metric,
            'weight': 2,
        },
        'SOD': {
            'func': f1_metric,
            'weight': 2,
        },
        'FLOE': {
            'func': f1_metric,
            'weight': 1,
        },
    },
    'num_val_scenes': 10,  # Number of scenes randomly sampled from train_list to use in validation.

    # -- GPU/cuda options -- #
    'gpu_id': 0,  # Index of GPU. In case of multiple GPUs.
    'num_workers': 2,  # Number of parallel processes to fetch data.
    'num_workers_val': 1,  # Number of parallel processes during validation.

    # -- U-Net Options -- #
    'unet_conv_filters': [16, 32, 64, 64],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.

}

device = torch.device(f"cuda:{train_options['gpu_id']}")

def initialise_model():

    # if torch.cuda.is_available():
    #     print(colour_str('GPU available!', 'green'))
    #     print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))

    #
    # else:
    #     print(colour_str('GPU not available.', 'red'))
    #     device = torch.device('cpu')


    print('Loading model.')
    # Setup U-Net model, adam optimizer, loss function and dataloader.
    net_vis = UNet(options=train_options).to(device)
    net_vis.load_state_dict(torch.load('best_model')['model_state_dict'])
    print('Model successfully loaded.')



    import glob
    train_options['train_list'] = glob.glob('C:/Users/Ultimate Gaming Comp/Documents/Ice_Data_2/*')
    np.random.seed(0)
    train_options['validate_list'] = np.random.choice(np.array(train_options['train_list']),
                                                      size=train_options['num_val_scenes'], replace=False)
    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list'] if
                                   scene not in train_options['validate_list']]
    #print(train_options['test_list'])


    #upload_package = xr.Dataset()  # To store model outputs.
    dataset = AI4ArcticChallengeDataset(files=train_options['validate_list'], options=train_options)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)
    print('Setup ready')

    print('Testing.')

    return net_vis, dataloader


def run_vis(model, data):

    os.makedirs('inference', exist_ok=True)
    model.eval()

    print('starting eval')



    for inf_x, _, masks, scene_name in tqdm(iterable=data, total=len(train_options['validate_list']), colour='green',
                                            position=0):


        scene_name = scene_name[:19]  # Removes the _prep.nc from the name.
        torch.cuda.empty_cache()
        inf_x = inf_x.to(device, non_blocking=True)

        with torch.no_grad(), torch.cuda.amp.autocast():
            inf_x = inf_x.to(device, non_blocking=True)
            output = model(inf_x)

        for chart in train_options['charts']:
            output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
            # upload_package[f"{scene_name}_{chart}"] = xr.DataArray(name=f"{scene_name}_{chart}",
            #                                                        data=output[chart].astype('uint8'),
            #                                                        dims=(f"{scene_name}_{chart}_dim0",
            #                                                              f"{scene_name}_{chart}_dim1"))

        # - Show the scene inference.
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        for idx, chart in enumerate(train_options['charts']):
            ax = axs[idx]
            output[chart] = output[chart].astype(float)
            output[chart][masks] = np.nan
            ax.imshow(output[chart], vmin=0, vmax=train_options['n_classes'][chart] - 2, cmap='jet',
                      interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

        plt.suptitle(f"Scene: {scene_name}", y=0.65)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
        fig.savefig(f"inference/{scene_name}.png", format='png', dpi=128, bbox_inches="tight")
        plt.close('all')

    # - Save upload_package with zlib compression.
    print('Saving upload_package. Compressing data with zlib.')
    #compression = dict(zlib=True, complevel=1)
    #encoding = {var: compression for var in upload_package.data_vars}
    #upload_package.to_netcdf('upload_package.nc', mode='w', format='netcdf4', engine='netcdf4', encoding=encoding)
    print('Testing completed.')

if __name__ == '__main__':
    model, data = initialise_model()
    run_vis(model, data)

