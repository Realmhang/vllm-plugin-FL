# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def _override_flashmla_sparse_backend():
    """强制覆盖 FLASHMLA_SPARSE backend 指向我们的自定义实现"""
    try:
        from vllm.attention.backends.registry import register_backend, AttentionBackendEnum
        
        # 重新注册 FLASHMLA_SPARSE，覆盖默认的 vLLM 原生实现
        register_backend(
            AttentionBackendEnum.FLASHMLA_SPARSE,
            class_path="vllm_fl.v1.attention.backends.mla.flashmla_sparse.MacaFlashMLASparseBackend",
        )
        
        # 同时注册 FLASHMLA
        register_backend(
            AttentionBackendEnum.FLASHMLA,
            class_path="vllm_fl.v1.attention.backends.mla.flashmla.MacaFlashMLABackend",
        )
        
        print("[vllm_fl] Successfully overridden FLASHMLA_SPARSE backend to use vllm_fl implementation")
    except Exception as e:
        print(f"[vllm_fl] Warning: Failed to override backend: {e}")

# 在模块导入时立即执行
_override_flashmla_sparse_backend()

########### platform plugin ###########
def register():
    """Register the FL platform."""
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()
    
    # Check if we're running on MetaX (Maca) platform
    # If so, also register MetaX-specific components
    if _is_metax_platform():
        logger.info("MetaX platform detected, registering MetaX-specific components")
        _register_metax_components()
    
    return "vllm_fl.device_platform.PlatformFL"


def _is_metax_platform() -> bool:
    """Detect if running on MetaX (Maca) platform."""
    try:
        # Check via vendor name from device info
        from vllm_fl.utils import DeviceInfo
        device_info = DeviceInfo()
        return device_info.vendor_name == "metax"
    except Exception:
        # Fallback: check environment variable or device properties
        try:
            import torch
            device_name = torch.cuda.get_device_name(0).lower()
            return "metax" in device_name or "maca" in device_name
        except Exception:
            return False


def _register_metax_components():
    """
    Register MetaX-specific components from vllm_fl.
    This ensures compatibility with vllm_fl functionality.
    """
    try:
        # Import and call vllm_fl register functions
        import vllm_fl
        
        # Register MetaX ops (includes patches)
        try:
            vllm_fl.register_ops()
            logger.info("Registered MetaX ops")
        except Exception as e:
            logger.warning(f"Failed to register MetaX ops: {e}")
        
        # Register MetaX models
        try:
            vllm_fl.register_model()
            logger.info("Registered MetaX models")
        except Exception as e:
            logger.warning(f"Failed to register MetaX models: {e}")
        
        # Register MetaX quantization configs
        try:
            vllm_fl.register_quant_configs()
            logger.info("Registered MetaX quantization configs")
        except Exception as e:
            logger.warning(f"Failed to register MetaX quant configs: {e}")
            
    except ImportError:
        logger.debug("vllm_fl not available, skipping MetaX component registration")
    except Exception as e:
        logger.warning(f"Error registering MetaX components: {e}")


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def register_ops():
    """Register FL ops."""
    import vllm_fl.ops  # noqa: F401
    
    # Also register MetaX ops if on MetaX platform
    if _is_metax_platform():
        try:
            import vllm_fl
            vllm_fl.register_ops()
        except Exception as e:
            logger.debug(f"Could not register MetaX ops: {e}")


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry
    import vllm.model_executor.models.qwen3_next as qwen3_next_module

    # Register Qwen3.5 MoE config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5_moe import Qwen3_5MoeConfig
        _CONFIG_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeConfig
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE config error: {str(e)}")

    # Register Qwen3Next model
    try:
        from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM  # noqa: F401

        qwen3_next_module.Qwen3NextForCausalLM = Qwen3NextForCausalLM
        logger.warning(
            "Qwen3NextForCausalLM has been patched to use vllm_fl.models.qwen3_next, "
            "original vLLM implementation is overridden"
        )

        ModelRegistry.register_model(
            "Qwen3NextForCausalLM",
            "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register Qwen3Next model error: {str(e)}")

    # Register Qwen3.5 MoE model
    try:
        ModelRegistry.register_model(
            "Qwen3_5MoeForConditionalGeneration",
            "vllm_fl.models.qwen3_5:Qwen3_5MoeForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE model error: {str(e)}")

    # Register MiniCPMO model
    try:
        ModelRegistry.register_model(
            "MiniCPMO",
            "vllm_fl.models.minicpmo:MiniCPMO"
        )
    except Exception as e:
        logger.error(f"Register MiniCPMO model error: {str(e)}")

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration",
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")

    # Register GLM-5 (GlmMoeDsa) model
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        glm5_model()

        ModelRegistry.register_model(
            "GlmMoeDsaForCausalLM",
            "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
    
    # Also register MetaX models if on MetaX platform
    if _is_metax_platform():
        try:
            import vllm_fl
            vllm_fl.register_model()
        except Exception as e:
            logger.debug(f"Could not register MetaX models: {e}")


def register_quant_configs():
    """Register quantization configs."""
    # FL-specific quant configs (if any) can be added here
    
    # Also register MetaX quant configs if on MetaX platform
    if _is_metax_platform():
        try:
            import vllm_fl
            vllm_fl.register_quant_configs()
        except Exception as e:
            logger.debug(f"Could not register MetaX quant configs: {e}")


# Backward compatibility: collect_env function
def collect_env() -> None:
    """Collect environment information."""
    try:
        from vllm_fl.collect_env import main as collect_env_main
        collect_env_main()
    except ImportError:
        logger.debug("vllm_fl.collect_env not available")