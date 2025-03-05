# Maintainer Guidelines

**This guide is for maintainers.** These people have **write
access** to NREL HPC's repository and help merge the contributions of
others.

If you have something to contribute, please see the [Contributing Guide](CONTRIBUTING.md).

This is a living document - if you see something out of date or missing,
speak up!

## What are a maintainer's responsibilities?

It is every maintainer's responsibility to:

* Expose a clear roadmap for improving the repository.
* Deliver prompt feedback and decisions on pull requests.
* Be available to anyone with questions, bug reports, criticism etc. on the repository.
  This includes GitHub issues and pull requests.
* Make sure the repository respects the philosophy, design and roadmap of the project.

## How are decisions made?

This project is an open-source project with an open design philosophy. This
means that the repository is the source of truth for EVERY aspect of the
project, including its philosophy, design, and roadmap. *If it's
part of the project, it's in the repo. It's in the repo, it's part of
the project.*

As a result, all decisions can be expressed as changes to the
repository. 

All decisions affecting this project, big and small, follow the same procedure:

1. Open a pull request.
   Anyone can do this.
2. Discuss the pull request.
   Anyone can do this.
3. Review the pull request.
   The relevant maintainers do this (see below [Who decides what?](#who-decides-what)).
   Changes that affect project management (changing policy, cutting releases, etc.) are [proposed and voted on](GOVERNANCE.md).
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

Overall the maintainer system works because of mutual respect.
The maintainers trust one another to act in the best interests of the project.
Sometimes maintainers can disagree and this is part of a healthy project to represent the points of view of various people.
In the case where maintainers cannot find agreement on a specific change, maintainers should use the [governance procedure](GOVERNANCE.md) to attempt to reach a consensus.

### How are maintainers added?

The best maintainers have a vested interest in the project.  Maintainers
are first and foremost contributors that have shown they are committed to
the long term success of the project.  Contributors wanting to become
maintainers are expected to be deeply involved in contributing code,
pull request review, and triage of issues in the project for more than two months.

Just contributing does not make you a maintainer, it is about building trust with the current maintainers of the project and being a person that they can depend on to act in the best interest of the project.
The final vote to add a new maintainer should be approved by the [governance procedure](GOVERNANCE.md).

### How are maintainers removed?

When a maintainer is unable to perform the [required duties](#what-are-a-maintainers-responsibilities) they can be removed by the [governance procedure](GOVERNANCE.md).
Issues related to a maintainer's performance should be discussed with them among the other maintainers so that they are not surprised by a pull request removing them.
