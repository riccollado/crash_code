# Copilot Commit Instructions

Create a commit message for the changes made on ALL staged files. The first line should
be a brief overall description of all changes. Then leave an empty line and add a
numbered list of changes. Include on the list every file that has a committed change. On
each list element, list first the file that it affects (in bold) and then the change.
For example:

1. **myfile.py**: Added a new function that does something.
2. **myfile2.py**: Removed a function that was not needed.

Make sure that only git-staged files are included in the changes list. If you are not
sure which files are staged, run `git status` to see the list of staged files. Also,
make sure that each staged file is included in the list of changes. If you are not sure
what changes were made to a file, run `git diff <filename>` to see the changes.
