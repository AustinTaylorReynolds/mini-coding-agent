# PR review checklist

Correctness
- Does the code do what the PR description / commit message says?
- Off-by-one errors, unhandled None/empty input, wrong comparison or operator?
- Are errors handled, or silently swallowed?

Tests
- Is every new branch or behaviour covered by a test?
- Do the tests assert behaviour, or only that nothing crashed?

Safety
- Any hard-coded secrets, unvalidated input, or shell/SQL built from user data?
- Any destructive operation (delete, overwrite) without a guard?

Maintainability
- Clear names, no dead code, no copy-pasted blocks, no unrelated changes mixed in.