"""
Training script for Wan2.2-S2V-14B with RenderMe360 dataset.
Modified from train.py to support S2V with audio processing.
"""

import torch, os, json
from torch.utils.data import DataLoader
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.renderme360_dataset import RenderMe360S2VDataset, passthrough_collate
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanS2VTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        audio_processor_path=None,  # NEW: for S2V audio processing
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

        # Parse audio processor config (NEW for S2V)
        audio_processor_config = None
        if audio_processor_path:
            parts = audio_processor_path.split(":")
            audio_processor_config = ModelConfig(
                model_id=parts[0],
                origin_file_pattern=parts[1] if len(parts) > 1 else ""
            )
            print(f"[S2V] Loading audio processor from: {audio_processor_path}")

        # Load pipeline with audio processor
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config  # NEW
        )

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Verify LoRA modules matched (CRITICAL CHECK)
        if lora_base_model:
            self._verify_lora_modules()

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _verify_lora_modules(self):
        """Verify that LoRA target modules actually matched some modules."""
        try:
            dit = self.pipe.dit
            module_names = [n for n, _ in dit.named_modules()]

            # Common LoRA target patterns
            target_patterns = ["to_q", "to_k", "to_v", "proj", "ff", "mlp", "ffn", "q_proj", "k_proj", "v_proj", "o_proj"]
            matched = [n for n in module_names if any(p in n for p in target_patterns)]

            print(f"[LoRA] Found {len(matched)} potential target modules in DiT")
            if len(matched) > 0:
                print(f"[LoRA] Sample matched modules: {matched[:5]}")
            else:
                print(f"[LoRA] WARNING: No LoRA target modules found! Check lora_target_modules setting.")
                print(f"[LoRA] Available module patterns: {set([n.split('.')[-1] for n in module_names[:20]])}")
        except Exception as e:
            print(f"[LoRA] Could not verify modules: {e}")


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

        # Extra inputs (CRITICAL: Use dataset values, don't override!)
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                # Use the separately loaded input_image from dataset, NOT video[0]
                inputs_shared["input_image"] = data["input_image"]
            elif extra_input == "input_audio":
                # S2V audio input
                inputs_shared["input_audio"] = data["input_audio"]
            elif extra_input == "audio_sample_rate":
                # S2V audio sample rate
                inputs_shared["audio_sample_rate"] = data["audio_sample_rate"]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                # Other extra inputs from dataset
                if extra_input in data:
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

    # Add S2V-specific argument
    parser.add_argument("--audio_processor_path", type=str, default=None,
                       help="Path to audio processor (e.g., 'Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/')")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Wan2.2-S2V Training with RenderMe360 Dataset")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_base_path}")
    print(f"Metadata: {args.dataset_metadata_path}")
    print(f"Resolution: {args.height}×{args.width}")
    print(f"Frames: {args.num_frames}")
    print(f"LoRA: {args.lora_base_model} (rank {args.lora_rank})" if args.lora_base_model else "Full fine-tuning")
    print(f"{'='*80}\n")

    # Create custom dataset
    dataset = RenderMe360S2VDataset(
        base_path=args.dataset_base_path,
        metadata_csv=args.dataset_metadata_path,
        cameras=["cam_28", "cam_37", "cam_49", "cam_54"],
        repeat=args.dataset_repeat
    )

    print(f"[Dataset] Total samples: {len(dataset)}")

    # Create training module
    model = WanS2VTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_path=args.audio_processor_path,  # NEW
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

    # Create model logger
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    # Launch training
    # Note: Default collate_fn=lambda x: x[0] in launch_training_task works fine for batch_size=1
    launch_training_task(
        dataset,
        model,
        model_logger,
        args=args
    )
