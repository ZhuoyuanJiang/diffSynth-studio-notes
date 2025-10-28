"""
Training script for RenderMe360 dataset with Wan2.2-S2V-14B.
Based on upstream train.py with modifications for RenderMe360UnifiedDataset.

Key differences from upstream:
1. Uses RenderMe360UnifiedDataset instead of UnifiedDataset
2. Fixed forward_preprocess to respect explicit input_image from dataset
3. Simplified dataset initialization (no operator map needed)
"""

import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.renderme360_unified_dataset import RenderMe360UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanRenderMe360TrainingModule(DiffusionTrainingModule):
    """
    Training module for RenderMe360 dataset.

    Inherits from upstream WanTrainingModule but fixes input_image handling
    to support datasets that provide explicit input_image (cam_28) separate
    from the target video (2x2 grid).
    """

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)

        # Enable VRAM management for 48GB GPUs (layer-wise CPU offloading)
        # Note: Cannot call enable_vram_management() here because pipe.device is "cpu" (DeepSpeed handles device placement)
        # VRAM management will be enabled by DeepSpeed's model parallelism instead
        # self.pipe.enable_vram_management()

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs
        # MODIFIED: Check if dataset provides explicit input_image before defaulting to video[0]
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                # If dataset explicitly provides input_image, use it
                # Otherwise fall back to video first frame (upstream behavior)
                if "input_image" in data and data["input_image"] is not None:
                    inputs_shared["input_image"] = data["input_image"]
                else:
                    inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # Use RenderMe360UnifiedDataset instead of UnifiedDataset
    # Much simpler - no need for operator map or data_file_keys
    print(f"[RenderMe360] Loading dataset from: {args.dataset_base_path}")
    print(f"[RenderMe360] Metadata: {args.dataset_metadata_path}")

    dataset = RenderMe360UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        cameras=["cam_28", "cam_37", "cam_49", "cam_54"],  # Grid order: TL, TR, BL, BR
    )

    print(f"[RenderMe360] Dataset loaded: {len(dataset)} total samples")

    model = WanRenderMe360TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
