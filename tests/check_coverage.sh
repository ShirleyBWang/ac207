 #!/bin/bash

a=$(sed -En 's@(^.*<td class="right" data-ratio=".* .*">)(.*)(%</td>.*$)@\2@gp' tests/htmlcov/index.html)
b=90 # minium coverage 
if (( ${a: -2} < b));
    then
    return -1
fi
