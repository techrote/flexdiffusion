# Bootstrap PR scope

This change set is intentionally limited to the first backend plumbing gate. It does not add UI controls or persist latent/image checkpoint artefacts yet.

Review focus:

- default-off behaviour;
- classic-backend-only capability routing;
- one-based schedule semantics;
- no retained CUDA tensor chain;
- incremental manifest validity;
- interrupted-run status;
- unit-test coverage.
