#!/usr/bin/env python

from setuptools import setup
import site, sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]
if __name__ == "__main__":
    setup(name='ADA-2022-PROJECT-HOTCHOCOLATE',
      version='0.0',
      description='ADAProject',
      author='HotChocolate',
      packages=['Function'])