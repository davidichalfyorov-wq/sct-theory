# Contributing

This is an active research project in theoretical physics. Contributions, questions, and discussion are welcome.

## How to participate

- **Questions and discussion:** Open a [GitHub Discussion](https://github.com/davidichalfyorov-wq/sct-theory/discussions) or an Issue.
- **Bug reports:** If you find an error in a derivation, a failing test, or a discrepancy with published literature, please open an Issue with as much detail as possible.
- **Code contributions:** Please open an Issue describing the proposed change before submitting a Pull Request. This ensures alignment with the project direction.

## Before submitting a PR

1. Run the linter: `ruff check analysis/sct_tools/`
2. Run the test suite: `python -m pytest analysis/ -x -q`
3. If your change affects a derivation or prediction, verify it passes all relevant layers of the [8-layer verification pipeline](README.md#verification-philosophy).

## Scope

The repository covers formal derivations, numerical verification, and analysis code for Spectral Causal Theory. Contributions that extend verification coverage, improve numerical precision, add literature cross-checks, or fix errors are especially valued.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).
