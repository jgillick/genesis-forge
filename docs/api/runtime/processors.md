# Processors

One processor reproduces one action manager's process. The runtime ships processors for
every built-in manager; a custom action manager supplies its own by subclassing
`ActionManagerProcessor` and naming it in its deployment contract.

See [custom action managers](../../guide/deployment.md#custom-action-managers) in
the guide.

::: genesis_forge_runtime.ActionManagerProcessor

::: genesis_forge_runtime.AffineProcessor
