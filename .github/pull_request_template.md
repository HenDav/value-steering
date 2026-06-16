<!-- Thanks for contributing! Please read CONTRIBUTING.md. -->

## Summary

<!-- What does this change and why? -->

## Checklist

- [ ] `make lint` and `make test` pass (CPU suite is vLLM-free).
- [ ] Added/updated tests (CPU for pure logic; a `@pytest.mark.gpu` test if a feature must fire on an accelerator).
- [ ] If it touches a bound vLLM internal: grounded against the pinned vLLM source, and `value-steer-compat` was run on a CUDA box (paste the result).
- [ ] Updated `CHANGELOG.md` under "Unreleased".
- [ ] No new orphan code — gaps are an explicit `raise NotImplementedError` at one located point.
