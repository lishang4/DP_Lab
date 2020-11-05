#!/bin/bash

workdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $workdir

echo "starting install dependency packages ..."
echo "please wait ..."

if [ -n "$(python -V 2>&1 | grep "Python 3")" ]
then
    pyv=""
else
    pyv="3"
fi

echo "$(pip$pyv install -r requirement.txt)"
