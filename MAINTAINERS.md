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

#### I'm a maintainer, should I make pull requests too?

Yes. Nobody should ever push to the repository directly. All changes should be
made through a pull request.

### Who decides what?

Only maintainers are counted towards the required
review. In addition, if a maintainer has created a pull request, they cannot
review it themselves (to ensure equal amounts of review for every pull
request, no matter who wrote it).


## Handling Pull Requests

### Acceptance Criteria:
* The PR complies with the [contribution guidelines](CONTRIBUTING.md)
* The PR targets the appropriate branch 
* The content is clear, free of typos, and relevant to NREL HPC users

### Process for Pull Requests: 
* PRs can be merged once they have been approved by at least 1 maintainer. 
* Any new PR should be assigned a reviewer within 2 weeks. The reviewer should give a response within one week of assignment. 
* As a result of their review, a maintainer can:
    1. Request changes in the Pull Request (which blocks merging until the changes are resolved)
    3. Approve the Pull Request
    4. Propose closing the Pull Request 
* Maintainers need to submit the review in Github so that it is clear when the review is complete.  
* If the only issues holding up a merge are trivial fixes (typos, syntax, etc.) and the author doesn't respond within 2 weeks to the requested changes or comments, the maintainers can make the necessary changes themselves and proceed with the merge process. 
* If the requested changes to a PR don't receive a response from the submitter within 1 month, a maintainer can choose to take over the PR and make necessary changes or it can be closed.
* If a PR is related to an issue, check whether the issue is completely resolved and can be closed. Comment with the PR number when closing the issue. Modify or comment on the issue if it is not entirely resolved.
* A PR can be closed if it does not meet the acceptance criteria or does not receive a response to requested changes within 1 month. To close a PR, bring up a discussion at a Maintainer's meeting to get consensus. 
* If closing a PR, leave a comment about why it was closed. 






