


from src.models.unet import Unet

{
    '_target_': 'src.models.unet.Unet',
    'dim': 64,
    'dim_mults': [1, 2, 4],
    'resnet_block_groups': 8,
    'double_conv_layer': True,
    'learned_variance': False,
    'learned_sinusoidal_cond': False,
    'learned_sinusoidal_dim': 16,
    'input_dropout': 0.0,
    'block_dropout': 0.0,
    'block_dropout1': 0.0,
    'attn_dropout': 0.0,
    'with_time_emb': True,
    'keep_spatial_dims': False,
    'outer_sample_mode': None,
    'upsample_dims': None,
    'init_kernel_size': 7,
    'init_padding': 3,
    'init_stride': 1,
    'name': 'UNetR',
    'verbose': True,
    'loss_function': 'mse',
    'num_conditional_channels': 0
}

kwargs = {
    "num_input_channels": 2,
    "num_output_channels": 1,
}



if __name__ == "__main__":

    # # construct the command.
    # command = ["python", "run.py", "mode=test.yaml"]
    # subprocess.run(command)


    print("hi")