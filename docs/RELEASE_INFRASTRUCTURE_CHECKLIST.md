# Release infrastructure checklist

Use this checklist only after v0.4 has reached `FEATURE_COMPLETE` and its final
validated source state has been integrated. It is not authorization to release
an intermediate commit.

1. Confirm the intended version in `pyproject.toml` and `CITATION.cff`, then
   rerun the public CI, package-build, documentation, CodeQL, and pre-commit
   workflows from the integration commit.
2. Enable GitHub Pages with **GitHub Actions** as the publishing source, then
   confirm a successful `Documentation` deployment and its public URL.
3. Install and authorize the Codecov GitHub App for this repository, set the
   repository variable `CODECOV_ENABLED` to `true`, and confirm an OIDC coverage
   upload from the final CI before adding a Codecov badge.
4. Confirm that the OpenSSF Scorecard workflow published a public result before
   adding its badge.
5. Create the approved GitHub release and immutable version tag only after the
   final release review. Do not create a release to obtain a badge.
6. In PyPI, configure a Trusted Publisher for `ericrosenn1/LitDataMatcher`,
   workflow `.github/workflows/publish-to-pypi.yml`, and the protected `pypi`
   environment. Then manually dispatch the workflow with the validated tag and
   explicit confirmation.
7. Enable the Zenodo GitHub integration only for the approved release, verify
   the deposited archive and DOI, then add the DOI badge.
8. Replace the branch-qualified workflow badge URLs with `branch=main`, add
   only badges backed by live public results, and validate every image and link.

Record the final workflow URLs, service URLs, release/tag/DOI values, and badge
audit in the release handoff.
