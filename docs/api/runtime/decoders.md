# Decoders

One decoder reproduces one action manager's decode. The runtime ships decoders for
every built-in manager; a custom action manager supplies its own by subclassing
`ManagerDecoder` and naming it in its deployment contract.

See [custom action managers](../../guide/deployment.md#custom-action-managers) in
the guide.

::: genesis_forge_runtime.ManagerDecoder

::: genesis_forge_runtime.AffineDecoder
