from yolo.serving import export_saved_model_lib_v2
from yolo.common import registry_imports
from official.core import exp_factory
from official.modeling import hyperparams

EXP = 'scaled_yolo'
CFG_PATH = ['yolo/configs/experiments/yolov4-csp/inference/640.yaml']
def main():
    params = exp_factory.get_exp_config(EXP)
    for config_file in CFG_PATH or []:
        params = hyperparams.override_params_dict(
            params, config_file, is_strict=True)
    params.validate()
    params.lock()
    export_saved_model_lib_v2.export(
        input_type = 'image_tensor',
        batch_size = None,
        input_image_size= [640, 640],
        params=params,
        checkpoint_path='checkpoint/',
        export_dir = 'export_dir',
        num_channels=3, 
    )


if __name__ == '__main__':
    main()