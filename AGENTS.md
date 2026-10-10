# Sample rendering

Unless the user gives different instructions for the current task, artistic
sample renders and feature-validation image renders use the Titan RTX only,
512 x 512 resolution, and generation seeds 2, 3, 4 and 5. Record any separate
noise seed too.

Before rendering, inspect active Codex chats for GPU work and the Titan's
processes/utilization. Do not render while another chat is using or reserving
the Titan; benchmark cooling pauses are still reserved time. Continue CPU/UI
work and check again later. Do not stop another chat's jobs or alter GPU clocks
or power limits to obtain the GPU.

These sample defaults do not override explicit hardware-benchmark instructions.
