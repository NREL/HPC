from distutils.core import setup, Extension

module1 = Extension('spam',
                    sources = ['spam.c'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a process - core mapping package',
       ext_modules = [module1])


##python setup.py build
##export PYTHONPATH=`pwd`/build/lib.linux-x86_64-3.7

