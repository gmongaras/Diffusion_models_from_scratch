import threading
from .compute_model_stats import compute_model_stats




def compute_model_stats_multiple():
    # Directory name to load in models
    dir_name = "models_res"
    
    # Model filenames and parameter names
    model_filenames = [
        "model_40e_50000s.pkl",
        "model_80e_100000s.pkl",
        "model_120e_150000s.pkl",
        "model_160e_200000s.pkl",
        "model_200e_250000s.pkl",
        "model_240e_300000s.pkl",
        "model_280e_350000s.pkl",
        "model_320e_400000s.pkl",
        "model_360e_450000s.pkl",
        "model_400e_500000s.pkl"
    ]
    model_params_filenames = [
        "model_params_40e_50000s.json",
        "model_params_80e_100000s.json",
        "model_params_120e_150000s.json",
        "model_params_160e_200000s.json",
        "model_params_200e_250000s.json",
        "model_params_240e_300000s.json",
        "model_params_280e_350000s.json",
        "model_params_320e_400000s.json",
        "model_params_360e_450000s.json",
        "model_params_400e_500000s.json"
    ]

    # GPU numbers for each model to calculate statistics for
    gpu_nums = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ]

    # Diffusion parameters
    step_size = 1
    DDIM_scale = 1
    corrected = True

    # Batch size and number of images to generate
    num_fake_imgs = 10000
    batchSize = 200

    # Path to filename to save the mean and variance to
    file_path = "eval/saved_stats/res/scale_1_step_1/"

    # Output filenames
    mean_filenames = [
        "fake_mean_50000s.npy",
        "fake_mean_100000s.npy",
        "fake_mean_150000s.npy",
        "fake_mean_200000s.npy",
        "fake_mean_250000s.npy",
        "fake_mean_300000s.npy",
        "fake_mean_350000s.npy",
        "fake_mean_400000s.npy",
        "fake_mean_450000s.npy",
        "fake_mean_500000s.npy"
    ]
    var_filenames = [
        "fake_var_50000s.npy",
        "fake_var_100000s.npy",
        "fake_var_150000s.npy",
        "fake_var_200000s.npy",
        "fake_var_250000s.npy",
        "fake_var_300000s.npy",
        "fake_var_350000s.npy",
        "fake_var_400000s.npy",
        "fake_var_450000s.npy",
        "fake_var_500000s.npy"
    ]

    assert len(model_filenames) == len(model_params_filenames) ==\
        len(gpu_nums) == len(mean_filenames) == len(var_filenames),\
        "List lengths must be equal"


    # Iterate over all data and start a thread
    # for each of the statistical computations
    for i in range(0, len(model_filenames)):
        model_filename = model_filenames[i]
        model_params_filename = model_params_filenames[i]
        gpu_num = gpu_nums[i]
        mean_filename = mean_filenames[i]
        var_filename = var_filenames[i]

        th = threading.Thread(
            target=compute_model_stats,
            args = (dir_name,
                        model_filename,
                        model_params_filename,
                        "gpu",
                        gpu_num,
                        num_fake_imgs,
                        batchSize,
                        step_size,
                        DDIM_scale,
                        corrected,
                        file_path,
                        mean_filename,
                        var_filename)
        )
        th.start()



if __name__ == "__main__":
    compute_model_stats_multiple()