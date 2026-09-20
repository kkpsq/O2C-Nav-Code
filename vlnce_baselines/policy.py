"""Policy module: initialization, FMM planning setup, eval config."""

from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.models.sam_point_extractor import SAMPointExtractor


def initialize_policy(trainer):
    """Initialize mapping module, SAM extractor, and FusionMapPolicy.

    EXACT COPY of _initialize_policy method body from trainer.py,
    replacing self.xxx with trainer.xxx.
    """
    # logger.info("start to initialize policy")
    trainer.segment_module = GroundedSAM(trainer.config, trainer.device)
    trainer.sam_extractor = SAMPointExtractor(trainer.segment_module, default_merge_threshold=30)
    trainer.mapping_module = Semantic_Mapping(trainer.config.MAP).to(trainer.device)
    trainer.mapping_module.eval()
    trainer.visualizer.update_map(trainer.mapping_module)

    trainer.policy = FusionMapPolicy(trainer.config, trainer.mapping_module.map_shape[0])
    trainer.policy.reset()


def set_eval_config(trainer):
    """Set evaluation configuration.

    EXACT COPY of _set_eval_config method body from trainer.py,
    replacing self.xxx -> trainer.xxx.
    """
    trainer.config.defrost()
    trainer.config.MAP.DEVICE = trainer.config.TORCH_GPU_ID
    trainer.config.MAP.HFOV = trainer.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
    trainer.config.MAP.AGENT_HEIGHT = trainer.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
    trainer.config.MAP.NUM_ENVIRONMENTS = trainer.config.NUM_ENVIRONMENTS
    trainer.config.MAP.RESULTS_DIR = trainer.config.RESULTS_DIR
    trainer.world_size = trainer.config.world_size
    trainer.local_rank = trainer.config.local_rank
    trainer.config.freeze()
