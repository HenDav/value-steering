# Security Policy

## Reporting a vulnerability

Please report security issues **privately** by email to **hen.davidov@stats.ox.ac.uk** rather than
opening a public issue. Include a description, affected version, and a reproduction if possible.
We aim to acknowledge reports within a few working days.

## Scope and dual-use note

`value-steer` is **safety-research tooling**: it steers a language model's decoding using a learned
value head (e.g. toward safer outputs, or to abstain). The same machinery is label-agnostic and can
be trained on any verifier. Use it responsibly and in line with the licenses and acceptable-use
policies of the models and datasets you apply it to. The trained value heads we publish are research
artifacts and carry no warranty of safety or fitness for any purpose.

## Supported versions

Security fixes target the latest released version. The plugin binds to specific vLLM internals and
is pinned to a behaviorally-validated vLLM span (see `value_steer/validated_versions.json`); running
outside that span is unsupported.
