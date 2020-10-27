from yolo.configs import backbones


@exp_factory.register_config_factory('darknet_imagenet')
def image_classification_imagenet_darknet() -> cfg.ExperimentConfig:
    """Returns a revnet config for image classification on imagenet."""
    train_batch_size = 4096
    eval_batch_size = 4096
    steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size

    config = cfg.ExperimentConfig(
        task=ImageClassificationTask(
            model=ImageClassificationModel(
                num_classes=1001,
                input_size=[224, 224, 3],
                backbone=backbones.Backbone(
                    type='darknet',
                    revnet=backbones.DarkNet(model_id="darknettiny")),
                norm_activation=common.NormActivation(norm_momentum=0.9,
                                                      norm_epsilon=1e-5),
                add_head_batch_norm=True),
            losses=Losses(l2_weight_decay=1e-5),
            train_data=DataConfig(input_path=os.path.join(
                IMAGENET_INPUT_PATH_BASE, 'train*'),
                                  is_training=True,
                                  global_batch_size=train_batch_size),
            validation_data=DataConfig(input_path=os.path.join(
                IMAGENET_INPUT_PATH_BASE, 'valid*'),
                                       is_training=False,
                                       global_batch_size=eval_batch_size)),
        trainer=cfg.TrainerConfig(
            steps_per_loop=steps_per_epoch,
            summary_interval=steps_per_epoch,
            checkpoint_interval=steps_per_epoch,
            train_steps=90 * steps_per_epoch,
            validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
            validation_interval=steps_per_epoch,
            optimizer_config=optimization.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd',
                    'sgd': {
                        'momentum': 0.9,
                    }
                },
                'learning_rate': {
                    'type': 'polynomial',
                    'polynomial': {
                        "initial_learning_rate": 0.1,
                        "end_learning_rate": 0.0001,
                        "power": 4.0,
                    }
                },
                'warmup': {
                    'type': 'linear',
                    'linear': {
                        'warmup_steps': 5 * steps_per_epoch,
                        'warmup_learning_rate': 0
                    }
                }
            })),
        restrictions=[
            'task.train_data.is_training != None',
            'task.validation_data.is_training != None'
        ])
    return config
