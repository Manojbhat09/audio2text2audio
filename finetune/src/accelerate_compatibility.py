"""
Compatibility layer for accelerate library to fix dispatch_batches parameter issue.
"""
import logging
from accelerate import Accelerator

logger = logging.getLogger(__name__)

def patch_accelerator():
    """Patch the Accelerator class to handle dispatch_batches parameter gracefully."""
    original_init = Accelerator.__init__
    
    def patched_init(self, *args, **kwargs):
        # Remove dispatch_batches if it exists (it's not supported in this version)
        if 'dispatch_batches' in kwargs:
            logger.warning("Removing unsupported 'dispatch_batches' parameter from Accelerator")
            kwargs.pop('dispatch_batches')
        
        # Call the original init
        return original_init(self, *args, **kwargs)
    
    # Apply the patch
    Accelerator.__init__ = patched_init
    logger.info("✓ Accelerator compatibility patch applied")

def apply_compatibility_patches():
    """Apply all necessary compatibility patches."""
    patch_accelerator()
    logger.info("✓ All compatibility patches applied successfully")
