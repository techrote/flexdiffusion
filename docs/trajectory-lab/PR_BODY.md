## Scope

Bootstrap the SD1.x/classic Easy Diffusion trajectory path without UI or latent persistence yet.

## Included

- default-off trajectory configuration;
- deterministic exact/range/percentage schedule parser;
- atomic manifest recorder scaffold;
- classic backend capability declaration;
- live-latent callback tap;
- interruption/error/completion metadata;
- unit tests and minimal CI;
- baseline and GPU validation documentation.

## Explicitly deferred

- latent/preview persistence;
- writer queues and storage budgets;
- timeline UI;
- exact resume and branching;
- K80/other Easy Diffusion backends.

## Merge gate

Require the lightweight trajectory unit workflow to pass. GPU validation is the next stage before persistence work is considered stable.
