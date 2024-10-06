Pypi
====

Preparation:

* increment version in `setup.py`

Commands for releasing on pypi.org:

```
  find -name "*~" -delete
  python setup.py clean
  python setup.py sdist upload
```


Github
======

Steps:

* start new release (version: `vX.Y.Z`)
* enter release notes, i.e., significant changes since last release
* upload `pymeka3-examples-X.Y.Z.tar.gz` previously generated with `setup.py`
* publish
