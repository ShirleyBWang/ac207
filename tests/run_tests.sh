#!/usr/bin/env bash
# File       : run_tests.sh
# Description: Test suite driver script

set -e

# list of test cases you want to run
tests=(
    test_AutoDiff.py
    test_Dual.py
    test_Elem.py
    test_FM.py
    test_Node.py
    test_RM.py
)

# Must add the module source path because we use `import cs107_package` in
# our test suite.  This is necessary if you want to test in your local
# development environment without properly installing the package.
export PYTHONPATH="$(pwd -P)/..":${PYTHONPATH}

driver="${@}"

# run the tests
${driver} ${tests[@]}


