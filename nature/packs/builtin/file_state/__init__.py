"""file_state Pack — ReadMemory-based file awareness.

Provides:
- ReadMemory initialization on frame open
- edit_read_first Gate: Block Edit if file not in read_memory
- read_state_persist Listener: emit READ_MEMORY_SET after Read
- write_state_persist Listener: emit READ_MEMORY_SET after Edit/Write
"""

from nature.packs.builtin.file_state.pack import install, file_state_pack

__all__ = ["install", "file_state_pack"]
