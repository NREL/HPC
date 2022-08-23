# Maintainer Guidelines

**This guide is for maintainers.** These people have **write
access** to NREL HPC's repository and help merge the contributions of
others.

If you have something to contribute, please see the [Contributing Guide](CONTRIBUTING.md).

This is a living document - if you see something out of date or missing,
speak up!

## What are a maintainer's responsibilities?

It is every maintainer's responsibility to:

* Deliver prompt feedback and decisions on pull requests.
* Be available to anyone with questions, bug reports, criticism etc. on the repository.
  This includes GitHub issues and pull requests.

## How are decisions made?

All decisions affecting this project, big and small, follow the same procedure:

1. Open a pull request.
   Anyone can do this.
2. Discuss the pull request.
   Anyone can do this.
3. Review the pull request.
   The relevant maintainers do this (see below [Who decides what?](#who-decides-what)).
4. Merge or close the pull request.
   The relevant maintainers do this.

### I'm a maintainer, should I make pull requests too?

Yes. Nobody should ever push to master directly. All changes should be
made through a pull request.

## Who decides what?

All decisions are pull requests, and the relevant maintainers make
decisions by accepting or refusing the pull request. Review and acceptance
by anyone is denoted by adding a comment in the pull request.
However, only currently listed `MAINTAINERS` are counted towards the required
two Reviews. In addition, if a maintainer has created a pull request, they cannot
count toward the two Review rule (to ensure equal amounts of review for every pull
request, no matter who wrote it).


### Reviewing and approving Pull Requests
Acceptance Criteria:
PR complies with contribution guidelines
Targets the right branch
Content is clear/free of typos
Content is relevant to NREL HPC users

Process:
Any new Pull request should receive a response within 2 weeks. 
PRs should be merged once they have the review comments addressed and get approved by at least 1 maintainer. 
If the only issues holding up a merge are trivial fixes (typos, syntax, etc.) and the author doesn't respond within 2 weeks, the maintainers can make the necessary changes themselves and proceed with the merge process. 
If a PR doesn't receive feedback from the submitter within 1 month, a maintainer can choose to take over the PR and make necessary changes or can be closed.
If a PR is related to an issue. Check whether the issue is fixed 
If a PR doesn't receive feedback from the submitter within 1 month, a maintainer can choose to take over the PR and make necessary changes or can be closed.
If a PR is related to an issue, check whether the issue is fixed and can be closed. If they fix an issue, the issue should be closed/or modified. 
Discuss in GH-pages meeting if PR should be closed/not merged in. If the content is not relevant/usefu  

Maintainer review responses:
ask for changes in the PR (this blocks merging until the comments are resolved)
approve the PR
bring up to maintainer's to close PR. 

