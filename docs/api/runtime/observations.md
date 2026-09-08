# Observation Assembler

Rebuilds the policy's input vector from readings you supply each tick, in the order
training used, with the same per-entry scaling and history stacking.

Supply raw readings: any scaling recorded in the bundle is applied here.

::: genesis_forge_runtime.ObservationAssembler
