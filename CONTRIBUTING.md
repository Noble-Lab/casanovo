# Contributing to Casanovo

First off, thank you for taking the time to contribute.

The following document provides guidelines for contributing to both the documentation and the code of Casanovo.  
**No contribution is too small!**
Even fixing a simple typo is genuinely appreciated.

At the same time, we aim to maintain a high-quality, sustainable codebase.
These guidelines help ensure that contributions are useful, maintainable, and respectful of reviewers' time.

## Before you start

If you're unsure whether something is already being worked on, or whether your idea fits within the project scope, **please open an issue or ask the maintainers first**.
This helps avoid duplicated work and ensures your effort has the highest impact.

## Contributing to the documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to generate and deploy our documentation. Most pages are written in Markdown.

The API and CLI documentation are generated automatically from code docstrings.  
The Code of Conduct, Release Notes, and this Contributing guide live in the repository root.

### Editing most documents

The easiest way to edit documentation is via the "Edit on GitHub" button on each page:

1. Click "Edit on GitHub"
2. Click the pencil icon to edit
3. Make your changes
4. Fill in a short description
5. Click "Propose Changes"

Alternatively, you can edit files locally in the `docs/` directory (see below).

## Contributing to the code

We welcome contributions to the Casanovo codebase—especially those addressing existing [issues](https://github.com/Noble-Lab/Casanovo/issues).

### Development workflow

Casanovo follows a standard GitHub workflow with a few important specifics:

1. Fork the repository
2. Clone your fork locally
3. **Create your branch from the `dev` branch** (not `main`):
   ```bash
   git checkout dev
   git checkout -b my_feature_branch
   ```
4. Make your changes
5. Commit and push to your fork
6. Open a Pull Request (PR) **targeting the `dev` branch**

> [!NOTE]
> The `main` branch is reserved for stable releases. All active development happens in `dev`.

## Pull request requirements

To keep the project maintainable and reviewer-friendly, all PRs must meet the following criteria.

All pull requests are automatically reviewed by the CodeRabbit AI tool.
Carefully go through its feedback and address all comments, either by updating your code or by clearly explaining why a suggestion does not apply.
Ignoring automated review feedback will prevent further review.

In addition, all PRs are automatically linted and tested.
Your contribution must pass all tests and conform to the expected code style.
Using `black` locally (see below) is strongly recommended to avoid unnecessary iteration.

Only once all automated checks pass and CodeRabbit feedback has been addressed you should request a maintainer review.
At that point, please tag **@bittremieux**.
PRs that are not yet ready will not be reviewed.

## Writing good pull requests

**Keep changes focused**

Pull requests should be small, self-contained, and easy to review.
Avoid combining unrelated changes in a single PR.
For example, refactoring, feature additions, and bug fixes are best submitted separately.
If a PR becomes too large or difficult to review, maintainers may ask you to split it into smaller parts.

**Testing expectations**

Contributions should be accompanied by appropriate tests whenever applicable.
New functionality should be covered by tests, and bug fixes should ideally include a test that reproduces the original issue.
Contributions must not break existing tests, and should not reduce test coverage.
Reliable, deterministic tests are preferred.

**Documentation requirements**

Code changes should be understandable and discoverable.
Public functions and classes must include clear docstrings, and any change in behavior should be reflected in the documentation.
For more complex logic, brief inline comments can help future contributors understand the intent of the code.

## Python code style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use [black](https://black.readthedocs.io/en/stable/) for formatting.

We strongly recommend setting up pre-commit hooks:

```bash
# Install dependencies
pip install black pre-commit

# Activate in repository
pre-commit install
```

This ensures formatting issues are caught before committing.

## Use of AI coding tools

Contributions created with the help of AI coding tools are welcome.
However:

**You are fully responsible for your contribution.**

This includes ensuring that the code is correct, relevant, and consistent with the rest of the codebase.
Contributions should not contain unreviewed code or blindly generated output.
Submissions that do not meet basic quality standards may be rejected.

## Review process and etiquette

Code review is a collaborative process aimed at improving both the contribution and the project as a whole.
Please be open to feedback and willing to iterate on your work.
Maintainers may request changes not only for correctness, but also for clarity, consistency, or long-term maintainability.

Similarly, feedback from maintainers is intended to be constructive and supportive.
Clear communication and responsiveness help ensure that contributions can be merged efficiently.

## What makes a good contribution

Strong contributions are those that clearly address a defined problem, are technically sound, and integrate naturally with the existing codebase.
They are tested, documented, and thoughtfully implemented.
Contributors are encouraged to review their own changes before submission and ensure that their PR is ready for review.

---

Thank you again for contributing to Casanovo!
