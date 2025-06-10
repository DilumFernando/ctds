#!/bin/bash

echo "Running unit tests..."
python3 -m unittest test.py

TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted."
    exit 1
else
    echo "✅ All tests passed!"
    exit 0
fi
