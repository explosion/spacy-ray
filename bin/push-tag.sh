#!/usr/bin/env bash

set -e

# Insist repository is clean
git diff-index --quiet HEAD
git checkout master
git pull origin master
git push origin master

version=$(grep "version = " setup.cfg)
version=${version/version = }
git tag "v$version"
git push origin "v$version"
